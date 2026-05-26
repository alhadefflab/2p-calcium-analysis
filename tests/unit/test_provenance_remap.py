"""
Unit tests for _remap_provenance_paths in pipeline.py.
Run with:  pytest tests/unit/test_provenance_remap.py
"""
from pipeline import _remap_provenance_paths


class TestRemapProvenancePaths:

    def test_simple_string_forward_slash(self):
        result = _remap_provenance_paths(
            '/old/path/file.npy', '/old/path', '/new/path')
        assert result == '/new/path/file.npy'

    def test_simple_string_no_match(self):
        result = _remap_provenance_paths(
            '/other/path/file.npy', '/old/path', '/new/path')
        assert result == '/other/path/file.npy'

    def test_flat_dict(self):
        obj = {'a': '/old/path/a.npy', 'b': '/old/path/b.npy'}
        result = _remap_provenance_paths(obj, '/old/path', '/new/path')
        assert result == {'a': '/new/path/a.npy', 'b': '/new/path/b.npy'}

    def test_nested_dict(self):
        obj = {
            'load_data': {
                'filenames': {
                    'z1': '/old/path/z1/file.npy',
                    'z2': '/old/path/z2/file.npy',
                }
            }
        }
        result = _remap_provenance_paths(obj, '/old/path', '/new/path')
        assert result['load_data']['filenames']['z1'] == '/new/path/z1/file.npy'
        assert result['load_data']['filenames']['z2'] == '/new/path/z2/file.npy'

    def test_non_string_values_pass_through(self):
        obj = {'count': 42, 'flag': True, 'nested': {'n': 3.14}}
        result = _remap_provenance_paths(obj, '/old', '/new')
        assert result == obj

    def test_list_of_paths(self):
        obj = ['/old/path/a.npy', '/old/path/b.npy', 99]
        result = _remap_provenance_paths(obj, '/old/path', '/new/path')
        assert result[0] == '/new/path/a.npy'
        assert result[1] == '/new/path/b.npy'
        assert result[2] == 99

    def test_windows_backslash_in_stored_path(self):
        stored = r'C:\old\data\file.npy'
        result = _remap_provenance_paths(
            stored, r'C:\old\data', r'C:\new\data')
        assert result == r'C:\new\data\file.npy'

    def test_forward_slash_variant_of_windows_path(self):
        # Provenance may store the forward-slash version on Windows
        stored = 'C:/old/data/file.npy'
        result = _remap_provenance_paths(
            stored, r'C:\old\data', r'C:\new\data')
        assert result == 'C:/new/data/file.npy'

    def test_empty_string_unchanged(self):
        result = _remap_provenance_paths('', '/old', '/new')
        assert result == ''

    def test_empty_dict_unchanged(self):
        result = _remap_provenance_paths({}, '/old', '/new')
        assert result == {}

    def test_empty_list_unchanged(self):
        result = _remap_provenance_paths([], '/old', '/new')
        assert result == []

    def test_mixed_depth_provenance_shape(self):
        # Realistic provenance structure with mixed types
        obj = {
            'output_dir': '/old/root/ZH537',
            'load_data': {
                'args': {'ch_dict': {'mc_ch': 'ch1', 'func_ch': 'ch2'}},
                'filenames': {
                    'z1': {'ch1': '/old/root/ZH537/z1/ch1.ome.tif',
                           'ch2': '/old/root/ZH537/z1/ch2.ome.tif'},
                }
            },
            'rigid_motion_correction': {
                'z1': {'filenames': {'ch2': '/old/root/ZH537/z1/func.mmap'},
                       'motion_correct_obj': '/old/root/ZH537/z1/mc_obj.pkl'}
            }
        }
        result = _remap_provenance_paths(obj, '/old/root', '/new/root')
        assert result['output_dir'] == '/new/root/ZH537'
        assert result['load_data']['filenames']['z1']['ch1'] == \
               '/new/root/ZH537/z1/ch1.ome.tif'
        assert result['rigid_motion_correction']['z1']['filenames']['ch2'] == \
               '/new/root/ZH537/z1/func.mmap'
        # Non-path strings should be unchanged
        assert result['load_data']['args']['ch_dict']['mc_ch'] == 'ch1'
