from dataclasses import dataclass, field
from typing import Dict, Self

import torch

from gaussian_splatting import GaussianModel
from reduced_3dgs.quantization import VectorQuantizer

from ..payload import Payload
from ..interframe import InterframeEncoderInitConfig, InterframeCodecContext, InterframeCodecInterface


@dataclass
class VQInterframeCodecConfig(InterframeEncoderInitConfig):
    """
    Configuration parameters for VQ-based inter-frame codec.

    This dataclass holds the initialization settings for vector quantization,
    including the number of clusters for each attribute type.
    """
    num_clusters: int = 256
    num_clusters_rotation_re: int = None
    num_clusters_rotation_im: int = None
    num_clusters_opacity: int = None
    num_clusters_scaling: int = None
    num_clusters_features_dc: int = None
    num_clusters_features_rest: list = field(default_factory=list)
    max_sh_degree: int = 3
    tol: float = 1e-6
    max_iter: int = 500


@dataclass
class VQInterframeCodecContext(InterframeCodecContext):
    """
    Context data for VQ-based inter-frame encoding/decoding.

    This dataclass holds the quantization state including codebooks
    and cluster IDs for each frame.
    """
    ids_dict: Dict[str, torch.Tensor]
    codebook_dict: Dict[str, torch.Tensor]
    max_sh_degree: int


@dataclass
class VQKeyframePayload(Payload):
    """
    Payload for VQ keyframe data.

    Contains the full codebook and cluster IDs for the first frame.
    """
    ids_dict: Dict[str, torch.Tensor]
    codebook_dict: Dict[str, torch.Tensor]
    max_sh_degree: int

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new VQKeyframePayload instance on the target device.
        """
        return VQKeyframePayload(
            ids_dict={k: v.to(device) for k, v in self.ids_dict.items()},
            codebook_dict={k: v.to(device) for k, v in self.codebook_dict.items()},
            max_sh_degree=self.max_sh_degree,
        )


@dataclass
class VQInterframePayload(Payload):
    """
    Payload for VQ interframe data.

    Contains only the changed cluster IDs for subsequent frames.
    The codebook is inherited from the keyframe context.

    Attributes:
        ids_mask_dict: Dict of boolean tensors indicating which ids changed for each key.
        ids_dict: Dict of only the changed ids values (sparse).
    """
    ids_mask_dict: Dict[str, torch.Tensor]
    ids_dict: Dict[str, torch.Tensor]

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new VQInterframePayload instance on the target device.
        """
        return VQInterframePayload(
            ids_mask_dict={k: v.to(device) for k, v in self.ids_mask_dict.items()},
            ids_dict={k: v.to(device) for k, v in self.ids_dict.items()},
        )


class VQInterframeCodecInterface(InterframeCodecInterface):
    """
    VQ-based inter-frame encoding/decoding interface.

    This interface uses vector quantization to compress Gaussian model attributes.
    The keyframe generates codebooks, and subsequent frames use the same codebooks
    to find nearest cluster IDs.
    """

    def decode_interframe(self, payload: VQInterframePayload, prev_context: VQInterframeCodecContext) -> VQInterframeCodecContext:
        """
        Decode a delta payload to reconstruct the next frame's context.

        Applies the changed values from the payload to the previous context.

        Args:
            payload: The delta payload containing changed cluster IDs with masks.
            prev_context: The context of the previous frame (contains codebook).

        Returns:
            The reconstructed context for the current frame.
        """
        # Clone previous ids_dict and apply changes
        new_ids_dict = {}
        for key, prev_ids in prev_context.ids_dict.items():
            new_ids = prev_ids.clone()
            mask = payload.ids_mask_dict[key]
            new_ids[mask] = payload.ids_dict[key]
            new_ids_dict[key] = new_ids

        return VQInterframeCodecContext(
            ids_dict=new_ids_dict,
            codebook_dict=prev_context.codebook_dict,
            max_sh_degree=prev_context.max_sh_degree,
        )

    def encode_interframe(self, prev_context: VQInterframeCodecContext, next_context: VQInterframeCodecContext) -> VQInterframePayload:
        """
        Encode the difference between two consecutive frames.

        Compares prev and next contexts to find changed rows and stores only
        the changed values with their corresponding masks.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A payload containing only changed cluster IDs with masks.
        """
        # Find changed ids for each key
        ids_mask_dict = {}
        changed_ids_dict = {}
        for key in next_context.ids_dict.keys():
            prev_ids = prev_context.ids_dict[key]
            next_ids = next_context.ids_dict[key]
            # Compare ids, handling different tensor shapes
            if prev_ids.dim() == 1:
                mask = prev_ids != next_ids
            else:
                mask = (prev_ids != next_ids).any(dim=-1)
            ids_mask_dict[key] = mask
            changed_ids_dict[key] = next_ids[mask]

        return VQInterframePayload(
            ids_mask_dict=ids_mask_dict,
            ids_dict=changed_ids_dict,
        )

    def decode_keyframe(self, payload: VQKeyframePayload) -> VQInterframeCodecContext:
        """
        Decode a keyframe payload to create initial context.

        Args:
            payload: The keyframe payload containing full codebook and IDs.

        Returns:
            The context for the first/key frame.
        """
        return VQInterframeCodecContext(
            ids_dict=payload.ids_dict,
            codebook_dict=payload.codebook_dict,
            max_sh_degree=payload.max_sh_degree,
        )

    def encode_keyframe(self, context: VQInterframeCodecContext) -> VQKeyframePayload:
        """
        Encode the first frame as a keyframe.

        Args:
            context: The context of the first frame.

        Returns:
            A payload containing the full codebook and IDs.
        """
        return VQKeyframePayload(
            ids_dict=context.ids_dict,
            codebook_dict=context.codebook_dict,
            max_sh_degree=context.max_sh_degree,
        )

    def keyframe_to_context(self, frame: GaussianModel, init_config: VQInterframeCodecConfig) -> VQInterframeCodecContext:
        """
        Convert a keyframe to a VQInterframeCodecContext.

        Creates a VectorQuantizer and generates codebooks from the frame.

        Args:
            frame: The GaussianModel frame to convert.
            init_config: Configuration parameters for quantization.

        Returns:
            The corresponding VQInterframeCodecContext representation.
        """
        quantizer = VectorQuantizer(
            num_clusters=init_config.num_clusters,
            num_clusters_rotation_re=init_config.num_clusters_rotation_re,
            num_clusters_rotation_im=init_config.num_clusters_rotation_im,
            num_clusters_opacity=init_config.num_clusters_opacity,
            num_clusters_scaling=init_config.num_clusters_scaling,
            num_clusters_features_dc=init_config.num_clusters_features_dc,
            num_clusters_features_rest=init_config.num_clusters_features_rest,
            max_sh_degree=init_config.max_sh_degree,
            tol=init_config.tol,
            max_iter=init_config.max_iter,
        )

        # Generate codebooks and cluster IDs from the keyframe
        ids_dict, codebook_dict = quantizer.quantize(frame, update_codebook=True)

        return VQInterframeCodecContext(
            ids_dict=ids_dict,
            codebook_dict=codebook_dict,
            max_sh_degree=frame.max_sh_degree,
        )

    def interframe_to_context(
        self,
        frame: GaussianModel,
        prev_context: VQInterframeCodecContext,
    ) -> VQInterframeCodecContext:
        """
        Convert a frame to a VQInterframeCodecContext using the previous context's codebook.

        Uses the codebook from the previous context to find nearest cluster IDs.

        Args:
            frame: The GaussianModel frame to convert.
            prev_context: The context from the previous frame.

        Returns:
            The corresponding VQInterframeCodecContext representation.
        """
        # Create a quantizer with the same settings
        quantizer = VectorQuantizer(
            max_sh_degree=prev_context.max_sh_degree,
        )
        # Set the codebook from previous context
        quantizer._codebook_dict = prev_context.codebook_dict

        # Find nearest cluster IDs using existing codebook
        ids_dict = quantizer.find_nearest_cluster_id(frame, prev_context.codebook_dict)

        return VQInterframeCodecContext(
            ids_dict=ids_dict,
            codebook_dict=prev_context.codebook_dict,
            max_sh_degree=prev_context.max_sh_degree,
        )

    def context_to_frame(self, context: VQInterframeCodecContext, frame: GaussianModel) -> GaussianModel:
        """
        Convert a VQInterframeCodecContext back to a GaussianModel frame.

        Dequantizes the cluster IDs using the codebook to reconstruct attributes.
        Does not modify xyz coordinates.

        Args:
            context: The VQInterframeCodecContext to convert.
            frame: An empty GaussianModel or one from previous pipeline steps.
                This frame will be modified in-place with the context data.

        Returns:
            The modified GaussianModel with the frame data.
        """
        # Create a quantizer for dequantization
        quantizer = VectorQuantizer(
            max_sh_degree=context.max_sh_degree,
        )

        # Dequantize to reconstruct the model (xyz=None means don't modify xyz)
        frame = quantizer.dequantize(
            frame,
            context.ids_dict,
            context.codebook_dict,
            xyz=None,
            replace=True,
        )

        return frame
