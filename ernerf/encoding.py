from .freqencoder import FreqEncoder
from .shencoder import SHEncoder
from .gridencoder import GridEncoder
from .ashencoder import AshEncoder

ENCODERS = {
    'None': None,
    'frequency': FreqEncoder,
    'spherical_harmonics': SHEncoder,
    'hashgrid': GridEncoder,
    'tiledgrid': GridEncoder,
    'ash': AshEncoder
}

def get_encoder(encoding, input_dim=3,
                multires=6,
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding not in ENCODERS:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, spherical_harmonics, hashgrid, tiledgrid]')

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    EncoderClass = ENCODERS[encoding]

    if encoding in ['hashgrid', 'tiledgrid']:
        encoder = EncoderClass(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype=encoding, align_corners=align_corners)
    elif encoding == 'ash':
        encoder = EncoderClass(input_dim=input_dim, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)
    else:
        encoder = EncoderClass(input_dim=input_dim, degree=degree)

    return encoder, encoder.output_dim