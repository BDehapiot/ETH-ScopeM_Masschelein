#%% Imports -------------------------------------------------------------------

from pylibCZIrw import czi as pyczi

#%% Function : open_czi() -----------------------------------------------------

def open_czi(paths):
    C1s, C2s = [], []
    for path in paths:
        with pyczi.open_czi(str(path)) as czidoc:
            C1 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}).squeeze()
            C2 = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 1}).squeeze()
        C1s.append(C1)
        C2s.append(C2)
    return C1s, C2s