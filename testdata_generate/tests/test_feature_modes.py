"""Feature-only ablations must preserve geometry, targets and storage contracts."""
from dataclasses import replace

import numpy as np
import pytest

from train_generate.config import DataConfig, GenerationConfig
from train_generate.features import FEATURE_CONTRACTS, feature_contract
from train_generate.generate import extract_grad9, generate_blueprints, generate_training_splits, sample_blueprint
from train_generate.io import save_training_dataset_hdf5, load_training_arrays_from_hdf5
from testdata_generate.config import TestDataConfig as FlowerConfig, FlowerScenario
from testdata_generate.generate import generate_test_data

CROSS = [1, 3, 4, 5, 7]
COLS19 = list(range(9)) + [9+i for i in CROSS] + [18+i for i in CROSS]


def test_cross_normal_geometry_and_legacy_compatibility():
    phi = np.random.default_rng(42).normal(size=(11, 11))
    indices = np.array([[4, 4], [6, 6]])
    full = extract_grad9(phi, indices)
    np.testing.assert_array_equal(extract_grad9(phi, indices, cross_only=True), full[:, CROSS])
    np.testing.assert_array_equal(extract_grad9(phi, indices, center_only=True), full[:, [4]])
    np.testing.assert_array_equal(extract_grad9(np.zeros_like(phi), indices, cross_only=True), np.zeros((2,5,2)))
    assert extract_grad9(phi, np.empty((0,2),dtype=int), cross_only=True).shape == (0,5,2)
    with pytest.raises(ValueError):
        extract_grad9(phi, indices, center_only=True, cross_only=True)


@pytest.mark.parametrize('scale_h,alpha', [(False,()), (True,()), (True,(.5,1.,2.))])
def test_cross_features_are_27d_subset_on_paired_circle_and_ellipse(scale_h, alpha):
    config = DataConfig(resolutions=(32,),variations=1,ellipse_num_a=2,ellipse_variations_per_a=1,
        grid_convention='cell_count_cell_centres',feature_mode='phi9_full_normal',
        scale_h=scale_h,augment_scale_alpha=alpha)
    blueprints = generate_blueprints(config)
    cross_config = replace(config,feature_mode='phi9_cross_normal')
    assert generate_blueprints(cross_config) == blueprints
    for shape in ('circle','ellipse'):
        blueprint = next(b for b in blueprints if b['meta']['shape_type']==shape)
        full = sample_blueprint(blueprint, data_config=config)
        cross = sample_blueprint(blueprint, data_config=cross_config)
        assert len(full)==len(cross)>0
        for a,b in zip(full,cross):
            assert a.keys()==b.keys()
            for key in a:
                np.testing.assert_array_equal(b[key], a[key][:,COLS19] if key=='features' else a[key])


def test_cross_training_hdf5_roundtrip(tmp_path):
    config = DataConfig(resolutions=(16,),variations=2,shape_types=('circle',),
        grid_convention='cell_count_cell_centres',scale_h=True,feature_mode='phi9_cross_normal')
    generation = GenerationConfig(num_workers=1)
    bundle = generate_training_splits(config,generation)
    path = save_training_dataset_hdf5(bundle,generation,path=tmp_path/'cross.h5')
    loaded = load_training_arrays_from_hdf5(path)
    assert loaded['config'].feature_mode=='phi9_cross_normal'
    assert loaded['config'].grid_convention==config.grid_convention
    assert loaded['split_blueprint_indices']==bundle['split_blueprint_indices']
    for split in bundle['splits']:
        assert loaded['splits'][split]['features'].shape[1]==19
        for key in ('phi9','features','hkappa_target'):
            np.testing.assert_array_equal(loaded['splits'][split][key],bundle['splits'][split][key])


@pytest.mark.parametrize('mode', list(FEATURE_CONTRACTS))
def test_normalization_checkpoint_and_optimizer_roundtrip(mode,tmp_path):
    import torch
    from evaluate.shared import (fit_feature_transform,apply_feature_transform,save_feature_stats,
        load_feature_transform,save_checkpoint_bundle,load_model_from_checkpoint,resolve_feature_transform)
    from model.config import MLP_TrainConfig
    from model.model import create_model
    from model.train import _validate_split
    version,dim,order=feature_contract(mode)
    features=np.random.default_rng(7).normal(size=(24,dim)).astype(np.float32)
    _validate_split('train',{'features':features,'phi9':features[:,:9],'hkappa_target':np.zeros((24,1))})
    ft=fit_feature_transform(features,dataset_path=tmp_path/'data.h5')
    assert (ft['feature_version'],ft['raw_feature_dim'],ft['feature_order'])==(version,dim,order)
    sidecar=tmp_path/'stats.csv';save_feature_stats(sidecar,transform=ft,dataset_path=tmp_path/'data.h5')
    restored=load_feature_transform(sidecar)
    np.testing.assert_array_equal(apply_feature_transform(features,restored),apply_feature_transform(features,ft))
    cfg=MLP_TrainConfig(input_dim=dim,raw_feature_dim=dim)
    model=create_model(cfg);x=torch.from_numpy(apply_feature_transform(features,ft))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    loss=model(x).square().mean();loss.backward();optimizer.step();model.eval()
    path=save_checkpoint_bundle(model,model_type='mlp',model_config=cfg,path=tmp_path/'model.pt',feature_transform=ft)
    loaded,meta=load_model_from_checkpoint(path,device=torch.device('cpu'))
    checkpoint_ft,_=resolve_feature_transform(model_path=path,explicit_path=None,checkpoint_meta=meta)
    assert checkpoint_ft['raw_feature_dim']==dim
    with torch.inference_mode():torch.testing.assert_close(loaded(x),model(x),rtol=0,atol=0)
    if dim==19:
        from evaluate.shared import validate_feature_transform
        with pytest.raises(ValueError):validate_feature_transform({**ft,'feature_order':'phi9+nx9+ny9'})


def test_flower_generation_and_loader_preserve_cross_subset(tmp_path):
    from evaluate.flower import load_flower_dataset
    scenario=FlowerScenario('smooth','smooth',64,.25,33,1/64,.05,.15,3)
    config=FlowerConfig(scenarios=(scenario,),config_source='custom',test_iters=(0,1),scale_h=True,
        sign_mode='frozen_phi0',feature_mode='phi9_full_normal')
    full_path=generate_test_data(config,output=tmp_path/'full.h5')
    cross_path=generate_test_data(replace(config,feature_mode='phi9_cross_normal'),output=tmp_path/'cross.h5')
    import h5py
    with h5py.File(full_path) as full,h5py.File(cross_path) as cross:
        assert cross.attrs['feature_version']==5
        assert cross.attrs['feature_dim_raw']==19
        assert cross.attrs['feature_order']=='phi9+nx_cross5+ny_cross5'
        for key in full:
            np.testing.assert_array_equal(cross[key][:],full[key][:,COLS19] if key=='features' else full[key][:])
    loaded=load_flower_dataset(cross_path)
    assert loaded['arrays']['features'].shape[1]==19


def test_local_normals_affine_sign_scale_and_zero():
    from train_generate.features import local_normal_features
    from train_generate.geometry_core import STENCIL_OFFSETS
    phi=(2*STENCIL_OFFSETS[:,0]+3*STENCIL_OFFSETS[:,1]+7)[None,:].astype(float)
    expected=np.array([[2,3,3,3,2,2]])/np.sqrt(13)
    np.testing.assert_allclose(local_normal_features(phi),expected,rtol=1e-7)
    np.testing.assert_array_equal(local_normal_features(-phi),-local_normal_features(phi))
    np.testing.assert_array_equal(local_normal_features(phi*4),local_normal_features(phi))
    np.testing.assert_array_equal(local_normal_features(np.ones((2,9))),np.zeros((2,6)))
    assert local_normal_features(np.empty((0,9))).shape==(0,6)


def test_local_generation_preserves_samples_and_rebuild(tmp_path):
    from train_generate.features import local_normal_features
    from train_generate.rebuild_local import rebuild
    import h5py
    config=DataConfig(resolutions=(16,),variations=2,shape_types=('circle',),
        grid_convention='cell_count_cell_centres',scale_h=True,feature_mode='phi9_full_normal')
    generation=GenerationConfig(num_workers=1)
    full=generate_training_splits(config,generation)
    local=generate_training_splits(replace(config,feature_mode='phi9_local_normal'),generation)
    source=save_training_dataset_hdf5(full,generation,path=tmp_path/'full.h5')
    rebuild(source,tmp_path/'local.h5')
    loaded=load_training_arrays_from_hdf5(tmp_path/'local.h5')
    assert loaded['config'].feature_mode=='phi9_local_normal'
    for split in full['splits']:
        a,b=full['splits'][split],local['splits'][split]
        for key in ('phi9','hkappa_target'):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['features'][:,:9],b['features'][:,:9])
        np.testing.assert_array_equal(b['features'][:,9:],local_normal_features(b['phi9']))
        np.testing.assert_array_equal(b['features'],loaded['splits'][split]['features'])


def test_local_flower_roundtrip(tmp_path):
    from evaluate.flower import load_flower_dataset
    from train_generate.features import local_normal_features
    scenario=FlowerScenario('smooth','smooth',64,.25,33,1/64,.05,.15,3)
    cfg=FlowerConfig(scenarios=(scenario,),config_source='custom',test_iters=(0,1),scale_h=True,
        sign_mode='frozen_phi0',feature_mode='phi9_local_normal')
    path=generate_test_data(cfg,output=tmp_path/'flower.h5')
    import h5py
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f['features'][:,9:],local_normal_features(f['phi9'][:]))
    load_flower_dataset(path)


@pytest.mark.parametrize('scale_h,alpha',[(False,()),(True,()),(True,(.5,1.,2.))])
def test_local_circle_ellipse_scaling_and_sign(scale_h,alpha):
    from train_generate.features import local_normal_features
    cfg=DataConfig(resolutions=(32,),variations=1,ellipse_num_a=2,ellipse_variations_per_a=1,
        grid_convention='cell_count_cell_centres',feature_mode='phi9_full_normal',
        scale_h=scale_h,augment_scale_alpha=alpha)
    for shape in ('circle','ellipse'):
        b=next(b for b in generate_blueprints(cfg) if b['meta']['shape_type']==shape)
        full=sample_blueprint(b,data_config=cfg)
        local=sample_blueprint(b,data_config=replace(cfg,feature_mode='phi9_local_normal'))
        assert len(full)==len(local)
        for a,c in zip(full,local):
            for key in a:
                if key!='features':np.testing.assert_array_equal(a[key],c[key])
            np.testing.assert_array_equal(a['features'][:,:9],c['features'][:,:9])
            np.testing.assert_array_equal(c['features'][:,9:],local_normal_features(c['phi9']))


def test_local_normal_footprint_ends_at_three_by_three():
    from train_generate.features import local_normal_features
    from train_generate.generate import extract_phi9
    rng=np.random.default_rng(31)
    field=rng.normal(size=(7,7));ids=np.array([[3,3]])
    expected=local_normal_features(extract_phi9(field,ids))
    changed=rng.normal(size=(7,7))*1e9
    changed[2:5,2:5]=field[2:5,2:5]
    np.testing.assert_array_equal(local_normal_features(extract_phi9(changed,ids)),expected)
    np.testing.assert_array_equal(local_normal_features(extract_phi9(field[2:5,2:5],np.array([[1,1]]))),expected)
