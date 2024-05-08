import json
import keras.layers
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler


from mynetworks import Patches, PatchEncoder, mlp
from mynetworks import load_and_prepare_data
from mynetworks import resize_images


from typing import Optional, Dict, List, Tuple
from mytypes import Filename, NDArray, Mask

layers = keras.layers
models = keras.models

class OverwriteError(ValueError):
    def __init__(self, old: dict, new: dict, keys: List):
        old_values: List = [old.get(key) for key in keys]
        new_values: List = [new.get(key) for key in keys]
        whitespaces: int = len('OverwriteError: ')
        indent = ' ' * whitespaces
        self.message = f'the following keys would be overwritten:\n{keys=}\n{old_values=}\n{new_values=}'
        super().__init__(self.message)


class Parameters(dict):
    def __init__(self, mapping: Optional[dict] = None, /, load: Optional[Filename] = None, **kwargs) -> None:
        """kwargs are added to dict and will overwrite the same keys in mapping"""
        if mapping is None:
            mapping = {}
        super().__init__(mapping)
        self.update(kwargs)
        if load is not None:
            self.update(self._load(load))

    def __add__(self, other: dict):
        """return new Parameters object adding other dict to self overwriting old arguments"""
        new = {**self, **other}
        return Parameters(new)

    def __iadd__(self, other: dict):
        self.add(other, overwrite=True)
        return self

    def _would_overwrite(self, other: dict, strict: bool = False) -> bool:
        """raises OverwriteError if adding other to self would overwrite (actually change) something
        if strict checks with 'is' otherwise with '=='"""
        commom_keys: set = set(other.keys()).intersection(self.keys())
        if strict:
            overwrite_keys: list = [key for key in commom_keys if self.get(key) is not other.get(key)]
        else:
            overwrite_keys: list = [key for key in commom_keys if self.get(key) != other.get(key)]
        if overwrite_keys:
            raise OverwriteError(self, other, overwrite_keys)
        return False

    def add(self, other: dict, overwrite: bool = False, strict: bool = False) -> None:
        """ add other dict to self
        :param other: dict to be added
        :param overwrite: determines whether parameters can be overwritten
        :param strict: passed to would_overwrite
        """
        if not overwrite: self._would_overwrite(other, strict=strict)
        self.update(other)

    @staticmethod
    def _load(filename: Filename) -> dict:
        """loads parameters from file"""
        with open(filename, 'r') as file:
            return json.load(file)

    def save(self, filename: Optional[Filename] = None) -> None:
        """write parameters to file"""
        if filename is None:
            filename = self['modeldir'] + self['modelname'] + '.json'
        with open(filename, 'w') as file:
            json.dump(self, file)
        print(f'INFO: params saved as {filename}')


def data_from_params(parameters: Parameters, selection:Optional[Mask] = None) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    data_tuple = load_and_prepare_data(datafile=parameters['dataframefile'], rechitfile=parameters['rechitfile'],
                                       test_fraction=parameters['test_split'], preselectionfile=parameters['preselectionfile'],
                                       use_log=parameters['use_log'],
                                       selection=selection)
    (x_train, y_train), (x_test, y_test) = data_tuple
    # if 'vit' in parameters['modelname'].lower():
    #     x_train = resize_images(x_train)
    #     x_test = resize_images(x_test)
    return (x_train, y_train), (x_test, y_test) 

def weights_from_params(parameters, test_set: bool = False, 
                        selection: Optional[Mask] = None) -> NDArray:
    weights_ = np.load(parameters['weightfile'])

    if selection is None:
        selection = np.ones(weights_.shape, dtype=bool)
    weights_ = weights_[selection]
    if test_set: 
        weights_ = get_test_slice(parameters, weights_)
    else:
        weights_ = get_training_slice(parameters, weights_)
    return weights_

def get_training_slice(parameters: Parameters, data: NDArray) -> NDArray:
    stop_idx: int = int( (1-parameters['test_split']) * len(data) )
    training_slice: slice = slice(0, stop_idx)
    return data[training_slice]

def get_test_slice(parameters: Parameters, data: NDArray) -> NDArray:
    start_idx: int = int( (1-parameters['test_split']) * len(data) )
    test_slice: slice = slice(start_idx, None)
    return data[test_slice]

def split_data(parameters: Parameters, data: NDArray) -> Tuple[NDArray, NDArray]:
    training = get_training_slice(parameters, data)
    test = get_test_slice(parameters, data)
    return training, test

def check_params_work_together(parameter_list: List[Parameters], 
                                allow: Optional[List[str]] = None,
                                optional_restrict: Optional[List[str]] = None
                                ) -> None:
    quantities = [
        'dataframefile',
        'rechitfile',
        'weightfile',
        'preselectionfile',
        'test_split',
        'use_log',
        ]
    
    # check allow and optional_restrict do not overlap
    if allow is not None and optional_restrict is not None:
        in_both = [thing in optional_restrict for thing in allow]
        if in_both.any():
            raise ValueError(f'{thing} must not be in allow and optional_restrict')
    
    if allow is not None:
        for thing in allow: 
            quantities.pop(thing)

    if optional_restrict is not None:
        quantities += optional_restrict
    

    first_values = np.array([parameter_list[0][key] for key in quantities])
    for i, parameter in enumerate(parameter_list):
        values = np.array([parameter[key] for key in quantities])
        matches: Mask = values == first_values
        if not (matches).all():
            wrong_iems = [quantities[idx] for idx in range(len(quantities)) if ~matches[i]]
            message = f'The following values in the {i+1}. Parameters do not match the first one: {wrong_iems}'
            raise AssertionError(message)


# # vit without eta/pt:
# def build_vit_from_params(parameters: Parameters) -> keras.Model:
#     num_patches = (parameters['image_size'] // parameters['patch_size']) ** 2

#     input_image = layers.Input(shape=parameters['input_shape'])
#     patches = Patches(parameters['patch_size'])(input_image)
#     encoded_patches = PatchEncoder(num_patches, parameters['projection_dim'])(patches)
    
#     for _ in range(parameters['transformer_layers']):
#         x1 = layers.LayerNormalization(epsilon=parameters['layer_norm_epsilon'])(encoded_patches)
#         attention_output = layers.MultiHeadAttention(
#             num_heads=parameters['num_heads'], key_dim=parameters['projection_dim'], dropout=parameters['dropout_rate']
#         )(x1, x1)
#         # Skip connection 1.
#         x2 = layers.Add()([attention_output, encoded_patches])
#         # Layer normalization 2.
#         x3 = layers.LayerNormalization(epsilon=parameters['layer_norm_epsilon'])(x2)
#         # MLP.
#         x3 = mlp(x3, hidden_units=parameters['transformer_units'], dropout_rate=parameters['dropout_rate'])
#         # Skip connection 2.
#         encoded_patches = layers.Add()([x3, x2])

#     # Create a [batch_size, projection_dim] tensor.
#     representation = layers.LayerNormalization(epsilon=parameters['layer_norm_epsilon'])(encoded_patches)
#     representation = layers.Flatten()(representation)
#     representation = layers.Dropout(parameters['final_dropout'])(representation)
#     # Add MLP.
#     features = mlp(representation, hidden_units=parameters['mlp_head_units'], dropout_rate=parameters['dropout_rate'])

#     outputs = layers.Dense(1, activation='sigmoid')(features)
#     model_ = keras.Model(inputs=input_image, outputs=outputs)
#     return model_


def build_vit_from_params(parameters: Parameters) -> keras.Model:
    num_patches = (parameters['image_size'] // parameters['patch_size']) ** 2

    input_image = layers.Input(shape=parameters['input_shape'])
    other_inputs = layers.Input(shape=(len(parameters['other_inputs']),))
    # input_pt = layers.Input(shape=(1,))
    # input_eta = layers.Input(shape=(1,))
    # input_pileup = layers.Input(shape=(1,))
    
    patches = Patches(parameters['patch_size'])(input_image)
    encoded_patches = PatchEncoder(num_patches, parameters['projection_dim'])(patches)
    
    for _ in range(parameters['transformer_layers']):
        x1 = layers.LayerNormalization(epsilon=parameters['layer_norm_epsilon'])(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=parameters['num_heads'], key_dim=parameters['projection_dim'], dropout=parameters['dropout_rate']
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=parameters['layer_norm_epsilon'])(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=parameters['transformer_units'], dropout_rate=parameters['dropout_rate'])
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=parameters['layer_norm_epsilon'])(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(parameters['final_dropout'])(representation)

    # add pt and eta an pileup Input
    representation = layers.Concatenate(name='Features')([representation, other_inputs])

    features = mlp(representation, hidden_units=parameters['mlp_head_units'], dropout_rate=parameters['dropout_rate'])

    outputs = layers.Dense(1, activation='sigmoid')(features)
    model_ = keras.Model(inputs=[input_image, other_inputs], outputs=outputs)
    return model_

def build_cnn_from_params(parameters: Parameters) -> models.Model:
    input_layer = layers.Input(shape=parameters['input_shape'], name='Input: image')
    other_inputs = layers.Input(shape=(len(parameters['other_inputs']),), name=f"Input: {parameters['other_inputs']}")
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = layers.Conv2D(64,(3, 3), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.concatenate(name='Features')([x, other_inputs])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(input_layer, output_layer)



def rescale(parameter: Parameters, data: NDArray, weights: NDArray) -> Tuple[NDArray, NDArray]:
    split_idx: int = int((1-parameter['test_split'])*len(data))
    train = data[:split_idx]
    test = data[split_idx:]
    if len(data.shape)==1:
        train = train.reshape(-1,1)
        test = test.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(train.reshape(-1,1), sample_weight=weights[:split_idx])
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    return scaled_train, scaled_test

def rescale_multiple(parameter: Parameters, data_list: List[NDArray], weights: NDArray
                     ) -> Tuple[List[NDArray], List[NDArray]]:
    train_list = []
    test_list = []
    for data in data_list:
        print(data.shape)
        train, test = rescale(parameter, data, weights)
        print(train.shape, test.shape)
        train_list += [train]
        test_list += [test]
    return train_list, test_list


if __name__=='__main__':  # do some testing
    a = Parameters({'x': 1}, y='bla')
    a.save('test.json')
    b = Parameters(load='test.json')

    print(a)
    print(b)
    print(type(b['x']))

    base = Parameters(
        ## data and modelnames
        modeldir='models/',
        modelname='vit',
        ModelName='ViT',
        dataframefile='data/new_data_pre.pkl',
        rechitfile='data/new_rechits_pre.npy',
        preselectionfile=None,
        test_split=0.2,
        use_log=True,
        reweight='weight',  # todo change to weights_fake/real

        ## for fitting
        fit_params=Parameters(
            validation_split=0.25, epochs=500, batch_size=4096, verbose=2),  # val_split0.25 means split of 60/20/20,
                                                                             # need to account for testsplit first
        learning_rate=1e-3,
        output_binwidth=0.05,

        ## build model stuff
        input_shape=[11, 11, 3],
        image_size=11,
        patch_size=4,
        # num_patches wird automatisch gesetzt
        projection_dim=16,
        num_heads=2,
        transformer_units=[16],  # =projection_dim
        transformer_layers=2,
        mlp_head_units=[32],
        dropout_rate=0.1,
        final_dropout=0.5,
        layer_norm_epsilon=1e-6,

        ## callbacks
        use_earlystopping=True,
        use_reduce_lr=False,
        use_checkpointing=False,
        early_stopping=Parameters(
            monitor='val_loss', min_delta=0, patience=25, mode='auto', verbose=2, restore_best_weights=True),
        reduce_lr=Parameters(
            monitor='val_loss', factor=0.9, patience=20, verbose=2),
        checkpointing=Parameters(monitor='val_accuracy', save_best_only=True),  # checkpointfile is defined in script
                                                                                # from other params
    )


    base.save('base.json')
    new = Parameters(load='base.json')
    print(new)
    print(new==base)
