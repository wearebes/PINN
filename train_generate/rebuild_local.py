"""Rebuild the strict 3x3 15D input from an existing canonical training HDF5.

Geometry, phi9, labels, ordering, splits and scaling are preserved verbatim.
Only features and their metadata change. Never overwrite a source or result.
"""
import argparse
import hashlib
from pathlib import Path
import h5py
import numpy as np
from train_generate.features import feature_contract, local_normal_features


def rebuild(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    version, dim, order = feature_contract('phi9_local_normal')
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as src, h5py.File(output, 'x') as dst:
        dst.attrs.update(dict(src.attrs))
        for name in src:
            if name not in ('train', 'val', 'test'):
                src.copy(name, dst)
                continue
            group = dst.create_group(name)
            group.attrs.update(dict(src[name].attrs))
            for key in src[name]:
                if key != 'features':
                    src[name].copy(key, group)
            old = src[name]['features']
            new = group.create_dataset('features', shape=(len(old),dim),dtype='float32',
                                       chunks=(min(8192,len(old)),dim),compression='gzip')
            for start in range(0,len(old),65536):
                stop=min(start+65536,len(old))
                phi=src[name]['phi9'][start:stop]
                new[start:stop]=np.concatenate((old[start:stop,:9],local_normal_features(phi)),axis=1)
            print(name,new.shape,flush=True)
        dst.attrs.update(feature_mode='phi9_local_normal', feature_version=version,
                         feature_dim_raw=dim, feature_order=order, augment_gradient=True,
                         dataset_name=output.name, output_dir=str(output.parent),
                         feature_source=str(source.resolve()),
                         normal_discretization='strict_phi9_centered_tangent_first_order_inward')
        alpha=str(src.attrs.get('augment_scale_alpha_json','[]')) != '[]'
        dst.attrs['feature_transform']=order+('_over_ah' if alpha else '_over_h' if src.attrs['scale_h'] else '')
    with source.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    with h5py.File(output,'r+') as dst:
        dst.attrs['feature_source_sha256']=digest
    print(output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rebuild(a.source,a.output)
