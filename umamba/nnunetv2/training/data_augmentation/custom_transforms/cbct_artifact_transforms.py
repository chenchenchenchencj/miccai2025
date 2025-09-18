from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np

class RandomMetalStreakTransform(AbstractTransform):
    """Approximate CBCT金属伪影: 在随机轴方向插入高强度条带。
    简化实现: 直接在若干随机切片/列加亮或加暗条纹。
    data: (C, Z, Y, X)
    """
    def __init__(self, p_per_sample=0.15, num_streaks=(1,3), intensity_range=(500,1500),
                 axes=(2,3), width=(1,4)):
        self.p_per_sample = p_per_sample
        self.num_streaks = num_streaks
        self.intensity_range = intensity_range
        self.axes = axes
        self.width = width

    def __call__(self, **data_dict):
        data = data_dict.get('data')
        if data is None:
            return data_dict
        for b in range(data.shape[0]):
            if np.random.uniform() > self.p_per_sample:
                continue
            n_streak = np.random.randint(self.num_streaks[0], self.num_streaks[1]+1)
            for _ in range(n_streak):
                axis = np.random.choice(self.axes)
                w = np.random.randint(self.width[0], self.width[1]+1)
                add_int = np.random.uniform(*self.intensity_range) * (1 if np.random.rand()<0.5 else -1)
                if axis == 2:  # along Y axis: choose X column(s)
                    x0 = np.random.randint(0, data.shape[3]-w+1)
                    data[b, :, :, x0:x0+w] += add_int
                elif axis == 3:  # along X axis: choose Y column(s)
                    y0 = np.random.randint(0, data.shape[2]-w+1)
                    data[b, :, y0:y0+w, :] += add_int
        data_dict['data'] = data
        return data_dict

class RandomSaturationClipTransform(AbstractTransform):
    """随机饱和裁剪: 将上/下百分位以外值裁剪，模拟过曝/欠曝."""
    def __init__(self, p_per_sample=0.2, lower_pct=(0.0, 2.0), upper_pct=(98.0, 100.0)):
        self.p_per_sample = p_per_sample
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def __call__(self, **data_dict):
        data = data_dict.get('data')
        if data is None:
            return data_dict
        for b in range(data.shape[0]):
            if np.random.uniform() > self.p_per_sample:
                continue
            lo = np.random.uniform(*self.lower_pct)
            hi = np.random.uniform(*self.upper_pct)
            lo_v = np.percentile(data[b], lo)
            hi_v = np.percentile(data[b], hi)
            np.clip(data[b], lo_v, hi_v, out=data[b])
        data_dict['data'] = data
        return data_dict

class RandomPoissonGaussianNoiseTransform(AbstractTransform):
    """Poisson + Gaussian 噪声组合."""
    def __init__(self, p_per_sample=0.15, lam_scale=(5,20), gauss_sigma=(5,25)):
        self.p_per_sample = p_per_sample
        self.lam_scale = lam_scale
        self.gauss_sigma = gauss_sigma

    def __call__(self, **data_dict):
        data = data_dict.get('data')
        if data is None:
            return data_dict
        for b in range(data.shape[0]):
            if np.random.uniform() > self.p_per_sample:
                continue
            img = data[b]
            # shift to positive
            min_v = img.min()
            shift = 0
            if min_v < 0:
                shift = -min_v + 1e-3
            tmp = img + shift
            lam = np.random.uniform(*self.lam_scale)
            # normalize to mean ~ lam
            mean_tmp = tmp.mean() + 1e-6
            scale = lam / mean_tmp
            tmp2 = np.random.poisson(tmp * scale) / scale
            # gaussian
            sigma = np.random.uniform(*self.gauss_sigma)
            tmp2 = tmp2 + np.random.normal(0, sigma, size=tmp2.shape)
            data[b] = tmp2 - shift
        data_dict['data'] = data
        return data_dict

