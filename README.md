# gsvvcompressor

A modular video compression framework for Gaussian Splatting models (3DGS). This library provides a flexible, extensible architecture for encoding and decoding sequences of `GaussianModel` frames using inter-frame compression techniques.

## Installation

Install from GitHub:

```sh
pip install git+https://github.com/yindaheng98/gsvvcompressor.git@master
```

Or clone and install locally:

```sh
git clone https://github.com/yindaheng98/gsvvcompressor.git
cd gsvvcompressor
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Dependencies are automatically installed (numpy, cloudpickle, zstandard, hydra-core, etc.)

## Quick Start

### Command-Line Interface

Encode a sequence of GaussianModel frames:

```sh
python -m gsvvcompressor encode vqxyzzstd \
    input.first_frame_path=data/frame_0000.ply \
    input.subsequent_format="data/frame_{:04d}.ply" \
    output.path=compressed.bin
```

Decode compressed data back to frames:

```sh
python -m gsvvcompressor decode vqxyzzstd \
    input.path=compressed.bin \
    output.first_frame_path=decoded/frame_0000.ply \
    output.subsequent_format="decoded/frame_{:04d}.ply"
```

### Programmatic Usage

```python
from gsvvcompressor.combinations import VQXYZZstdEncoder, VQXYZZstdDecoder
from gsvvcompressor.vq import VQInterframeCodecConfig
from gsvvcompressor.xyz import XYZQuantInterframeCodecConfig
from gsvvcompressor.io import FrameReader, FrameWriter, BytesReader, BytesWriter

# Create encoder
encoder = VQXYZZstdEncoder(
    vq_config=VQInterframeCodecConfig(num_clusters=256),
    xyz_config=XYZQuantInterframeCodecConfig(alpha=0.2),
    zstd_level=7,
)

# Read frames and encode
frame_reader = FrameReader(
    first_frame_path="data/frame_0000.ply",
    subsequent_format="data/frame_{:04d}.ply",
    start_index=1,
)
encoded_stream = encoder.encode_stream(frame_reader.read())

# Write compressed output
bytes_writer = BytesWriter("compressed.bin")
bytes_writer.write(encoded_stream)

# Decode
decoder = VQXYZZstdDecoder()
bytes_reader = BytesReader("compressed.bin")
decoded_stream = decoder.decode_stream(bytes_reader.read())

# Write decoded frames
frame_writer = FrameWriter(
    first_frame_path="decoded/frame_0000.ply",
    subsequent_format="decoded/frame_{:04d}.ply",
    start_index=1,
)
frame_writer.write(decoded_stream)
```

## Architecture Overview

The library uses a layered, modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GaussianModel Stream                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AbstractEncoder / AbstractDecoder               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  InterframeEncoder / InterframeDecoder                        │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  CombinedInterframeCodecInterface                       │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────────────────┐   │  │  │
│  │  │  │ XYZ Codec       │  │ VQ Codec                    │   │  │  │
│  │  │  │ (coordinates)   │  │ (attributes)                │   │  │  │
│  │  │  └─────────────────┘  └─────────────────────────────┘   │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              Payload                                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                AbstractSerializer / AbstractDeserializer            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  ZstdSerializer / ZstdDeserializer                            │  │
│  │  (cloudpickle + length-prefix framing + zstd compression)     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Bytes Stream                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Abstract Classes

### Payload

`Payload` is the abstract base class for intermediate data structures that flow between the encoding/decoding stages and the serialization/deserialization stages.

```python
from gsvvcompressor import Payload

@dataclass
class MyPayload(Payload):
    data: torch.Tensor

    def to(self, device) -> Self:
        return MyPayload(data=self.data.to(device))
```

### AbstractEncoder and AbstractDecoder

These are the top-level abstract classes that define the encoding/decoding interface. They use a two-stage process:

**Encoding Pipeline:**
1. `pack(frame)` → Converts `GaussianModel` to `Payload`
2. `serialize_frame(payload)` → Converts `Payload` to `bytes`

**Decoding Pipeline:**
1. `deserialize_frame(data)` → Converts `bytes` to `Payload`
2. `unpack(payload)` → Converts `Payload` to `GaussianModel`

```python
from gsvvcompressor import AbstractEncoder, AbstractDecoder

class MyEncoder(AbstractEncoder):
    def pack(self, frame: GaussianModel) -> Iterator[Payload]:
        # Convert frame to payload(s)
        yield MyPayload(...)

    def flush_pack(self) -> Iterator[Payload]:
        # Flush any buffered payloads
        return
        yield

class MyDecoder(AbstractDecoder):
    def unpack(self, payload: Payload) -> Iterator[GaussianModel]:
        # Convert payload to frame(s)
        yield reconstructed_frame

    def flush_unpack(self) -> Iterator[GaussianModel]:
        # Flush any buffered frames
        return
        yield
```

### AbstractSerializer and AbstractDeserializer

These classes handle the conversion between `Payload` objects and `bytes`.

```python
from gsvvcompressor import AbstractSerializer, AbstractDeserializer

class MySerializer(AbstractSerializer):
    def serialize_frame(self, payload: Payload) -> Iterator[bytes]:
        yield serialized_bytes

    def flush(self) -> Iterator[bytes]:
        yield remaining_bytes

class MyDeserializer(AbstractDeserializer):
    def deserialize_frame(self, data: bytes) -> Iterator[Payload]:
        yield deserialized_payload

    def flush(self) -> Iterator[Payload]:
        yield remaining_payloads
```

## Inter-frame Compression

The `interframe` module provides infrastructure for inter-frame compression, where frames are encoded as differences from previous frames.

### InterframeCodecInterface

This is the core abstract interface for implementing inter-frame codecs. It defines methods for:

- **Keyframe encoding/decoding**: First frame contains full data
- **Interframe encoding/decoding**: Subsequent frames contain only differences
- **Context management**: Converting between frames and codec-specific contexts

```python
from gsvvcompressor.interframe import (
    InterframeCodecInterface,
    InterframeCodecContext,
    InterframeEncoderInitConfig,
)

class MyCodecInterface(InterframeCodecInterface):
    def keyframe_to_context(self, frame: GaussianModel, init_config) -> MyContext:
        """Convert a keyframe to codec context (encoder only)."""
        ...

    def interframe_to_context(self, frame: GaussianModel, prev_context) -> MyContext:
        """Convert a frame using previous context as reference (encoder only)."""
        ...

    def encode_keyframe(self, context: MyContext) -> MyKeyframePayload:
        """Encode context as keyframe payload."""
        ...

    def encode_interframe(self, prev_context, next_context) -> MyInterframePayload:
        """Encode difference between two contexts."""
        ...

    def decode_keyframe(self, payload: MyKeyframePayload) -> MyContext:
        """Decode keyframe payload to context."""
        ...

    def decode_interframe(self, payload, prev_context) -> MyContext:
        """Decode interframe payload using previous context."""
        ...

    def context_to_frame(self, context: MyContext, frame: GaussianModel) -> GaussianModel:
        """Convert context back to frame (decoder only)."""
        ...
```

### InterframeEncoder and InterframeDecoder

These classes implement `AbstractEncoder` and `AbstractDecoder` using an `InterframeCodecInterface`:

```python
from gsvvcompressor.interframe import InterframeEncoder, InterframeDecoder

# Create encoder with a codec interface
encoder = InterframeEncoder(
    serializer=my_serializer,
    interface=my_codec_interface,
    init_config=my_init_config,
)

# Create decoder
decoder = InterframeDecoder(
    deserializer=my_deserializer,
    interface=my_codec_interface,
)
```

### CombinedInterframeCodecInterface

This class allows combining multiple `InterframeCodecInterface` instances into a single codec. Each sub-codec processes different aspects of the frame data:

```python
from gsvvcompressor.interframe.combine import CombinedInterframeCodecInterface

# Combine XYZ (coordinates) and VQ (attributes) codecs
combined = CombinedInterframeCodecInterface([
    xyz_interface,  # Handles xyz coordinates
    vq_interface,   # Handles other attributes (rotation, scaling, etc.)
])
```

## Built-in Codec Implementations

### VQ (Vector Quantization) Codec

The VQ codec uses vector quantization to compress Gaussian model attributes (rotation, opacity, scaling, features). It maintains a codebook generated from the keyframe.

```python
from gsvvcompressor.vq import VQInterframeCodecInterface, VQInterframeCodecConfig

vq_config = VQInterframeCodecConfig(
    num_clusters=256,              # Default clusters for each attribute
    num_clusters_rotation_re=256,  # Clusters for rotation real part
    num_clusters_rotation_im=256,  # Clusters for rotation imaginary part
    num_clusters_opacity=256,      # Clusters for opacity
    num_clusters_scaling=256,      # Clusters for scaling
    num_clusters_features_dc=256,  # Clusters for DC features
    max_sh_degree=3,               # Maximum SH degree
    tol=1e-6,                      # K-means tolerance
    max_iter=500,                  # K-means max iterations
)

vq_interface = VQInterframeCodecInterface()
```

**How it works:**
- **Keyframe**: Runs K-means clustering to generate codebooks, stores codebooks + cluster IDs
- **Interframe**: Uses existing codebooks to find nearest cluster IDs, stores only changed IDs with masks

### XYZ Quantization Codec

The XYZ codec quantizes the 3D coordinates of Gaussian splats using uniform quantization with adaptive step sizes.

```python
from gsvvcompressor.xyz import XYZQuantInterframeCodecInterface, XYZQuantInterframeCodecConfig

xyz_config = XYZQuantInterframeCodecConfig(
    k=1,                  # K-th nearest neighbor for step size estimation
    sample_size=10000,    # Points to sample for NN estimation
    seed=42,              # Random seed
    quantile=0.05,        # Quantile of NN distances for scale estimation
    alpha=0.2,            # Scaling factor for step size
    min_step=None,        # Optional minimum step size
    max_step=None,        # Optional maximum step size
    tolerance=0,          # Tolerance for change detection
)

xyz_interface = XYZQuantInterframeCodecInterface()
```

**How it works:**
- **Keyframe**: Computes quantization config (step_size, origin) from point distribution, stores config + quantized coordinates
- **Interframe**: Uses existing config to quantize coordinates, stores only changed coordinates with masks

### Zstd Serializer

The Zstd serializer uses cloudpickle for serialization with zstandard compression:

```python
from gsvvcompressor.compress.zstd import ZstdSerializer, ZstdDeserializer

serializer = ZstdSerializer(level=7)  # Compression level 1-22
deserializer = ZstdDeserializer()
```

## Pre-built Combinations

### VQXYZZstd

A ready-to-use encoder/decoder combining VQ + XYZ quantization + Zstd compression:

```python
from gsvvcompressor.combinations import (
    VQXYZZstdEncoder,
    VQXYZZstdDecoder,
    VQXYZZstdEncoderConfig,
    VQXYZZstdDecoderConfig,
)

# Direct construction
encoder = VQXYZZstdEncoder(
    vq_config=VQInterframeCodecConfig(...),
    xyz_config=XYZQuantInterframeCodecConfig(...),
    zstd_level=7,
    payload_device="cpu",
)

decoder = VQXYZZstdDecoder(payload_device="cpu")

# Or from config
from gsvvcompressor.combinations import build_vqxyzzstd_encoder, build_vqxyzzstd_decoder

encoder_config = VQXYZZstdEncoderConfig(
    vq=VQInterframeCodecConfig(...),
    xyz=XYZQuantInterframeCodecConfig(...),
    zstd_level=7,
)
encoder = build_vqxyzzstd_encoder(encoder_config)
```

## Registry System

The `combinations` module provides a registry for encoder/decoder combinations:

```python
from gsvvcompressor.combinations import (
    ENCODERS,
    DECODERS,
    register_encoder,
    register_decoder,
)

# List available codecs
print(ENCODERS.keys())  # ['vqxyzzstd', ...]
print(DECODERS.keys())  # ['vqxyzzstd', ...]

# Register a custom codec
register_encoder(
    name="mycodec",
    factory=build_my_encoder,
    config_class=MyEncoderConfig,
    description="My custom encoder",
)
```

## IO Module

The `io` module provides utilities for reading and writing frames and bytes:

```python
from gsvvcompressor.io import (
    FrameReader,
    FrameWriter,
    BytesReader,
    BytesWriter,
)

# Read GaussianModel frames
reader = FrameReader(
    first_frame_path="data/frame_0000.ply",
    subsequent_format="data/frame_{:04d}.ply",
    start_index=1,
    sh_degree=3,
)
for frame in reader.read():
    process(frame)

# Write GaussianModel frames
writer = FrameWriter(
    first_frame_path="output/frame_0000.ply",
    subsequent_format="output/frame_{:04d}.ply",
    start_index=1,
)
writer.write(frame_iterator)

# Read/write bytes
bytes_reader = BytesReader("compressed.bin", chunk_size=65536)
bytes_writer = BytesWriter("output.bin")
```

## Creating a Custom Codec

To create a custom inter-frame codec:

### 1. Define Payload Classes

```python
from dataclasses import dataclass
from gsvvcompressor import Payload

@dataclass
class MyKeyframePayload(Payload):
    # Full frame data
    data: torch.Tensor

    def to(self, device):
        return MyKeyframePayload(data=self.data.to(device))

@dataclass
class MyInterframePayload(Payload):
    # Delta data only
    mask: torch.Tensor
    delta: torch.Tensor

    def to(self, device):
        return MyInterframePayload(
            mask=self.mask.to(device),
            delta=self.delta.to(device),
        )
```

### 2. Define Context and Config

```python
from gsvvcompressor.interframe import InterframeCodecContext, InterframeEncoderInitConfig

@dataclass
class MyCodecConfig(InterframeEncoderInitConfig):
    quality: int = 10

@dataclass
class MyCodecContext(InterframeCodecContext):
    encoded_data: torch.Tensor
    quality: int
```

### 3. Implement the Interface

```python
from gsvvcompressor.interframe import InterframeCodecInterface

class MyCodecInterface(InterframeCodecInterface):
    def keyframe_to_context(self, frame, init_config):
        # Encode frame to context
        return MyCodecContext(
            encoded_data=encode(frame, init_config.quality),
            quality=init_config.quality,
        )

    def interframe_to_context(self, frame, prev_context):
        # Encode frame using previous context
        return MyCodecContext(
            encoded_data=encode(frame, prev_context.quality),
            quality=prev_context.quality,
        )

    def encode_keyframe(self, context):
        return MyKeyframePayload(data=context.encoded_data)

    def encode_interframe(self, prev_context, next_context):
        # Compute delta
        mask = prev_context.encoded_data != next_context.encoded_data
        delta = next_context.encoded_data[mask]
        return MyInterframePayload(mask=mask, delta=delta)

    def decode_keyframe(self, payload):
        return MyCodecContext(encoded_data=payload.data, quality=0)

    def decode_interframe(self, payload, prev_context):
        new_data = prev_context.encoded_data.clone()
        new_data[payload.mask] = payload.delta
        return MyCodecContext(encoded_data=new_data, quality=prev_context.quality)

    def context_to_frame(self, context, frame):
        # Decode context to frame
        frame._xyz = decode(context.encoded_data)
        return frame
```

### 4. Build the Encoder/Decoder

```python
from gsvvcompressor.interframe import InterframeEncoder, InterframeDecoder
from gsvvcompressor.compress.zstd import ZstdSerializer, ZstdDeserializer

def build_my_encoder(config):
    return InterframeEncoder(
        serializer=ZstdSerializer(level=7),
        interface=MyCodecInterface(),
        init_config=config,
    )

def build_my_decoder(config):
    return InterframeDecoder(
        deserializer=ZstdDeserializer(),
        interface=MyCodecInterface(),
    )
```

### 5. Register (Optional)

```python
from gsvvcompressor.combinations import register_encoder, register_decoder

register_encoder("mycodec", build_my_encoder, MyCodecConfig, "My custom codec")
register_decoder("mycodec", build_my_decoder, MyCodecConfig, "My custom codec")
```

## Project Structure

```
gsvvcompressor/
├── __init__.py              # Core exports: Payload, AbstractEncoder/Decoder, AbstractSerializer/Deserializer
├── __main__.py              # CLI entry point
├── payload.py               # Payload abstract base class
├── encoder.py               # AbstractEncoder base class
├── decoder.py               # AbstractDecoder base class
├── serializer.py            # AbstractSerializer base class
├── deserializer.py          # AbstractDeserializer base class
├── combinations/            # Pre-built codec combinations
│   ├── __init__.py
│   ├── registry.py          # Encoder/decoder registry
│   └── vq_xyz_zstd.py       # VQ + XYZ + Zstd combination
├── compress/                # Compression implementations
│   ├── __init__.py
│   └── zstd.py              # Zstd serializer/deserializer
├── interframe/              # Inter-frame compression infrastructure
│   ├── __init__.py
│   ├── interface.py         # InterframeCodecInterface and related classes
│   ├── encoder.py           # InterframeEncoder
│   ├── decoder.py           # InterframeDecoder
│   ├── combine.py           # CombinedInterframeCodecInterface
│   └── twopass.py           # Two-pass encoding utilities
├── io/                      # Input/output utilities
│   ├── __init__.py
│   ├── bytes.py             # BytesReader/Writer
│   ├── config.py            # Configuration dataclasses
│   └── gaussian_model.py    # FrameReader/Writer
├── vq/                      # Vector quantization codec
│   ├── __init__.py
│   ├── interface.py         # VQInterframeCodecInterface
│   └── twopass.py           # Two-pass VQ utilities
└── xyz/                     # XYZ coordinate codec
    ├── __init__.py
    ├── interface.py         # XYZQuantInterframeCodecInterface
    ├── quant.py             # Quantization functions
    └── ...                  # Other XYZ utilities
```

## Class Hierarchy

```
Payload (ABC)
├── VQKeyframePayload
├── VQInterframePayload
├── XYZQuantKeyframePayload
├── XYZQuantInterframePayload
└── CombinedPayload

AbstractEncoder (ABC)
└── InterframeEncoder

AbstractDecoder (ABC)
└── InterframeDecoder

AbstractSerializer (ABC)
└── ZstdSerializer

AbstractDeserializer (ABC)
└── ZstdDeserializer

InterframeCodecInterface (ABC)
├── VQInterframeCodecInterface
├── XYZQuantInterframeCodecInterface
└── CombinedInterframeCodecInterface

InterframeEncoderInitConfig
├── VQInterframeCodecConfig
├── XYZQuantInterframeCodecConfig
└── CombinedInterframeEncoderInitConfig

InterframeCodecContext
├── VQInterframeCodecContext
├── XYZQuantInterframeCodecContext
└── CombinedInterframeCodecContext
```

## License

See [LICENSE](LICENSE) for details.
