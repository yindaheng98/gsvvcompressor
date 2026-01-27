from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

from gaussian_splatting import GaussianModel

from ..encoder import AbstractEncoder
from ..payload import Payload
from ..serializer import AbstractSerializer
from .interface import (
    InterframeCodecContext,
    InterframeCodecInterface,
    InterframeEncoderInitConfig,
)


@dataclass
class PassOneContext:
    """
    Context data from the first pass of two-pass encoding.

    This dataclass holds the information gathered during the first pass
    over all frames, which is then used to optimize the encoding in the
    second pass. For example, this could contain statistics about the
    entire video sequence, codebook data, or other global information.

    Subclasses should define specific fields for their encoding scheme.
    """
    pass


class TwoPassInterframeCodecInterface(InterframeCodecInterface):
    """
    Abstract interface for two-pass inter-frame encoding/decoding algorithms.

    This interface extends InterframeCodecInterface to support two-pass encoding,
    where the first pass gathers information about all frames, and the second
    pass uses this information to optimize encoding.

    Design Guidelines:
        - PassOneContext: Store information gathered during the first pass
          (e.g., global statistics, codebook data, quality parameters).
        - The keyframe_to_context method receives PassOneContext instead of
          InterframeEncoderInitConfig, allowing it to use first-pass data.
    """

    @abstractmethod
    def pass_one(
        self,
        init_config: InterframeEncoderInitConfig,
        frames: List[GaussianModel],
    ) -> PassOneContext:
        """
        Perform the first pass over all frames to gather encoding information.

        This method processes all frames to collect statistics or other
        information needed for optimized encoding in the second pass.

        Args:
            init_config: Encoder initialization configuration.
            frames: List of all GaussianModel frames to be encoded.

        Returns:
            PassOneContext containing information gathered from the first pass.
        """
        pass

    @abstractmethod
    def keyframe_to_context(
        self,
        frame: GaussianModel,
        pass_one_context: PassOneContext,
    ) -> InterframeCodecContext:
        """
        Convert a keyframe to a Context using first-pass information.

        This method is called by the encoder when processing the first frame
        in the second pass. The pass_one_context provides information gathered
        during the first pass for optimized encoding.

        Args:
            frame: The GaussianModel frame to convert.
            pass_one_context: Context from the first pass containing
                global encoding information.

        Returns:
            The corresponding Context representation.
        """
        pass
