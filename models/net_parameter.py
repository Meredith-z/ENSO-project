from collections import OrderedDict
from Convcell import CGRU_cell,CLSTM_cell
from Predcell import PredCell
batch_size = 32
# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]})
    ],

    [
        CLSTM_cell(shape=(20,50),input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(10,25),input_channels=64, filter_size=5, num_features=64)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'deconv2_leaky_1': [64, 32, 3, 1, 1],
            'conv3_leaky_1': [32, 1, 1, 1, 0]
        })
        # OrderedDict({
        #     'conv2_leaky_1': [64, 16, 3, 1, 1],
        #     'conv3_leaky_1': [16, 1, 1, 1, 0]
        # }),
    ],

    [
        CLSTM_cell(shape=(10,25),input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(20,50),input_channels=64, filter_size=5, num_features=64)
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({
            'conv1_leaky_1': [1, 16, 3, 1, 1],
            'dropout1':[0.2]
        }),
        OrderedDict({
            'conv2_leaky_1': [32, 64, 3, 1, 1],
            'dropout2':[0.2]
                     }),

    ],

    [
        CGRU_cell(shape=(20,50), input_channels=16, filter_size=5, num_features=32),
        CGRU_cell(shape=(20,50), input_channels=64, filter_size=5, num_features=64)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({
            'deconv1_leaky_1': [64, 64, 3, 1, 1],
            'dropout2':[0.2]
        }),

        OrderedDict({
            'deconv2_leaky_1': [32, 1, 1, 1, 0]
            # 'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(20,50), input_channels=64, filter_size=5, num_features=64),
        CGRU_cell(shape=(20,50), input_channels=64, filter_size=5, num_features=32)
    ]
]

cnn_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 64, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]})
    ]
]

cnn_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'deconv2_leaky_1': [64, 32, 3, 1, 1],
            'conv3_leaky_1': [32, 1, 1, 1, 0]
        })
    ]
]

convlstm_encoder = [
    [
        CLSTM_cell(shape=(20,50),input_channels=1, filter_size=5, num_features=64),
        CLSTM_cell(shape=(20,50),input_channels=64, filter_size=5, num_features=64)
    ]
]

convlstm_decoder = [

    [
        CLSTM_cell(shape=(20,50),input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(20,50),input_channels=64, filter_size=5, num_features=64)
    ]
]

predrnn_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]})
    ],

    [
        PredCell(shape=(20,50),input_channels=16, filter_size=5, num_features=64),
        PredCell(shape=(10,25),input_channels=64, filter_size=5, num_features=64)
    ]
]

predrnn_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'deconv2_leaky_1': [64, 32, 3, 1, 1],
            'conv3_leaky_1': [32, 1, 1, 1, 0]
        })
        # OrderedDict({
        #     'conv2_leaky_1': [64, 16, 3, 1, 1],
        #     'conv3_leaky_1': [16, 1, 1, 1, 0]
        # }),
    ],

    [
        PredCell(shape=(10,25),input_channels=64, filter_size=5, num_features=64),
        PredCell(shape=(20,50),input_channels=64, filter_size=5, num_features=64)
    ]
]
