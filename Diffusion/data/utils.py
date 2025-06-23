import yaml, os, requests
from tqdm import tqdm
from pathlib import Path

def str_to_indices(string):

    """Expects a string in the format '32-123, 230-321'"""

    assert not string.endwith(','), f"Provided string '{string}' end with a comma, pls remove it"
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsets = sub.split("-")
        assert len(subsets) > 0
        if len(subsets) == 1:
            indices.append(int(subsets[0]))

        else:
            rang = [j for j in rang(int(subsets[0]), int(subsets[1]))]
            indices.extend(rang)

    return sorted(indices)



def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)

    for idx in indices:
        synsets.append(str(di2s[idx]))

    print(f"Using {len(synsets)} different synsets for construction of Restricted Imagenet.")

    return synsets



def download(url,
             local_path,
             chunk_size=1024):
    
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)



def retrieve(list_or_dict,
             key,
             splitval="/",
             default=None,
             expand=True,
             pass_success=False):
    

    """ 
    Given a nested list or dict return the desired value at key expanding 
    callable nodes if necessary and :attr: `expand` of `True`. The expansion
    is done in-place 

    Parameters
    ----------
        list_or_dict: list or dict 
            Possibly nested list or dictionary

        key: str 
            key/to/value, path like string describing all keys neccessary to 
            consider to get to the desired value. list indices can also be 
            passed here.

        splitval: str 
            string that defines the delimiter between keys of the 
            different depth levels in `key`.

        default: obj 
            value returned if :attr: `key` is not found.

        expand: bool 
            whether to expand callable nodes on the path or not.


    Returns
    --------
        The desired value or if :attr: `default` is not `None` and the 
        :attr: `key` is not found returns `default`.

    Raises
    ------
        Exception if `key` not in `list_or_dict` and :attr: `default` is None

        
    """


    keys = key.split(splitval)

    success = True

    try:
        visited = []
        parent = None 
        last_key = None 

        if key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable mode with expand=False."
                        ),
                        keys=keys,
                        visited=visited
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key 
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]

                else:
                    list_or_dict = list_or_dict[int(key)]

            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)
            

            visited += [key]

        # final expansion of retrived value 
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict

        

    except Exception as e:

        if default is None:
            raise e 
        
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    
    else:
        return list_or_dict, success
        




class KeyNotFoundError(Exception):

    def __init__(self,
                 cause,
                 keys=None,
                 visited=None):
        
        self.cause = cause
        self.keys = keys 
        self.visited = visited

        messages = list()
        if keys is not None:
            messages.append(f"key not found: {keys}")

        if visited is not None:
            messages.append(f"Visited: {visited}")

        messages.append(f"Cause: \n{cause}")
        message = "\n".join(messages)
        super().__init__(message)


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()

def mark_prepared(root):
    return Path(root).joinpath(".ready").touch()