
######## TASK DEFINITIONS ########


from datetime import datetime
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Optional, Any, ClassVar

import fibsem.config as fcfg
from fibsem import acquire, alignment, calibration
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import BeamType, FibsemImage, ImageSettings

from autolamella.structures import AutoLamellaStage, Lamella
from autolamella.workflows.core import (
    AutoLamellaUI,
    log_status_message,
    set_images_ui,
    update_milling_ui,
    update_status_ui,
)
from autolamella.protocol.validation import TRENCH_KEY

@dataclass
class AutoLamellaTaskConfig(ABC):
    """Configuration for AutoLamella tasks."""
    supervise: bool = True
    imaging: ImageSettings = field(default_factory=ImageSettings)
    milling: Dict[str, List[FibsemMillingStage]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        ddict = asdict(self)
        ddict["imaging"] = self.imaging.to_dict()
        ddict["milling"] = {k: [stage.to_dict() for stage in v] for k, v in self.milling.items()}
        return ddict

    @classmethod
    def from_dict(cls, ddict: Dict[str, Any]) -> 'AutoLamellaTaskConfig':
        kwargs = {}


        for f in fields(cls):
            if f.name in ddict:
                kwargs[f.name] = ddict[f.name]

        # unroll the parameters dictionary
        if "parameters" in ddict:
            for key, value in ddict["parameters"].items():
                if key in cls.__annotations__:
                    kwargs[key] = value
                else:
                    # QUERY: should we raise an error here or just ignore unknown parameters?
                    raise ValueError(f"Unknown parameter '{key}' in task configuration.")

        if "imaging" in ddict:
            kwargs["imaging"] = ImageSettings.from_dict(ddict["imaging"])
        if "milling" in ddict:
            kwargs["milling"] = {
                k: [FibsemMillingStage.from_dict(stage) for stage in v]
                for k, v in ddict["milling"].items()
            }

        return cls(**kwargs)

class AutoLamellaTask(ABC):
    """Base class for AutoLamella tasks."""
    task_name: ClassVar[str]
    display_name: ClassVar[str]
    config: AutoLamellaTaskConfig

    def __init__(self, 
                 microscope: FibsemMicroscope,
                 config: AutoLamellaTaskConfig, 
                 lamella: Lamella, 
                 parent_ui: Optional['AutoLamellaUI'] = None):
        self.microscope = microscope
        self.config = config
        self.lamella = lamella
        self.parent_ui = parent_ui
        self.task_id = str(uuid.uuid4())

    @abstractmethod
    def run(self) -> Lamella:
        """Run the task and return the updated lamella."""
        pass

@dataclass
class MillTrenchTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillTrenchTask."""
    align_reference: bool = False  # whether to align to a trench reference image
    charge_neutralisation: bool = True
    orientation: str = "FIB"


class MillTrenchTask(AutoLamellaTask):
    """Task to mill the trench for a lamella."""
    task_name: ClassVar[str] = "MILL_TRENCH"
    display_name: ClassVar[str] = "Mill Trench"
    config: MillTrenchTaskConfig

    def run(self) -> None:

        # TODO: make the pre-task and post-task updates more generic, so they can be reused in other tasks
        """Run the task to mill the trench for a lamella."""

        # pre-task
        self.lamella.state.stage = AutoLamellaStage.MillTrench # TODO: migrate to this self.task_id)
        self.lamella.state.start_timestamp = datetime.timestamp(datetime.now())
        log_status_message(self.lamella, "STARTED")

        # bookkeeping
        validate = self.config.supervise
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        log_status_message(self.lamella, "MOVE_TO_TRENCH")
        update_status_ui(self.parent_ui, f"{self.lamella.info} Moving to Trench Position...")
        trench_position = self.microscope.get_target_position(self.lamella.stage_position, 
                                                              self.config.orientation)
        self.microscope.safe_absolute_stage_movement(trench_position)

        # align to reference image
        # TODO: support saving a reference image when selecting the trench from minimap
        reference_image_path = os.path.join(self.lamella.path, "ref_PositionReady.tif")
        if os.path.exists(reference_image_path) and self.config.align_reference:
            log_status_message(self.lamella, "ALIGN_TRENCH_REFERENCE")
            update_status_ui(self.parent_ui, f"{self.lamella.info} Aligning Trench Reference...")
            ref_image = FibsemImage.load(reference_image_path)
            alignment.multi_step_alignment_v2(microscope=self.microscope, 
                                            ref_image=ref_image, 
                                            beam_type=BeamType.ION, 
                                            alignment_current=None,
                                            steps=1, subsystem="stage")

        log_status_message(self.lamella, "MILL_TRENCH")

        # get trench milling stages
        milling_stages = self.config.milling[TRENCH_KEY]

        # acquire reference images
        image_settings.hfw = milling_stages[0].milling.hfw
        image_settings.filename = f"ref_{self.lamella.status}_start"
        image_settings.save = True
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)
        set_images_ui(self.parent_ui, eb_image, ib_image)
        update_status_ui(self.parent_ui, f"{self.lamella.info} Preparing Trench...")
        
        # define trench milling stage
        milling_stages = update_milling_ui(self.microscope, milling_stages, self.parent_ui,
            msg=f"Press Run Milling to mill the trenches for {self.lamella.name}. Press Continue when done.",
            validate=validate,
        )

        # log the protocol
        self.config.milling = {TRENCH_KEY: deepcopy(milling_stages)}

        # charge neutralisation
        if self.config.charge_neutralisation:
            log_status_message(self.lamella, "CHARGE_NEUTRALISATION")
            update_status_ui(self.parent_ui, f"{self.lamella.info} Neutralising Sample Charge...")
            image_settings.beam_type = BeamType.ELECTRON
            calibration.auto_charge_neutralisation(self.microscope, image_settings)

        # reference images
        log_status_message(self.lamella, "REFERENCE_IMAGES")
        reference_images = acquire.take_set_of_reference_images(
            microscope=self.microscope,
            image_settings=image_settings,
            hfws=(fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH),
            filename=f"ref_{self.lamella.status}_final",
        )
        set_images_ui(self.parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

        # post-task
        self.lamella.state.microscope_state = self.microscope.get_microscope_state()
        self.lamella.state.end_timestamp = datetime.timestamp(datetime.now())
        log_status_message(self.lamella, "FINISHED")
        update_status_ui(self.parent_ui, f"{self.lamella.info} Finished")

        # we don't need both
        self.lamella.history.append(deepcopy(self.lamella.state))
        self.lamella.states[self.lamella.workflow] = deepcopy(self.lamella.state)
        self.lamella.tasks[self.task_name] = deepcopy(self.config)

# Lamella.tasks =  List[AutoLamellaTaskConfig]