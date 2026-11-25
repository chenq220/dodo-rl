"""Categorical mask encoding shared by recurrent execution and training."""
import numpy as np


def action_mask_vector(space, mask):
    """Flatten named categorical masks in the public head order."""
    categories = space.layout['category_counts']
    if not categories:
        if mask is not None:
            raise ValueError('Continuous MAPPO does not accept categorical masks')
        return None
    if mask is None:
        return np.ones(sum(categories), dtype=bool)
    if space.kind == 'multi_discrete':
        if set(mask) != set(space.heads):
            raise ValueError('MAPPO mask head names do not match action space')
        vectors = []
        for (name, _), count in zip(space.heads.items(), categories):
            vector = np.ones(count, dtype=bool) if mask[name] is None else np.asarray(mask[name], dtype=bool)
            if vector.shape != (count,):
                raise ValueError(f'MAPPO mask shape mismatch for head {name}')
            vectors.append(vector)
        return np.concatenate(vectors)
    vector = np.asarray(mask, dtype=bool)
    if vector.shape != (sum(categories),):
        raise ValueError('MAPPO action mask shape mismatch')
    return vector
