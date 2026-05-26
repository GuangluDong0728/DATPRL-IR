from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_16_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import math
import random
from torch.utils.data.sampler import Sampler
# from torch.utils.data import Sampler
# import random

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class MultiDatasetPairedImageDataset(data.Dataset):
    """Multi-dataset paired image dataset for ALL-IN-ONE image restoration.
    
    Supports loading from 6 different datasets simultaneously with data augmentation
    and intelligent batch sampling to ensure each dataset is represented.
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        datasets (list): List of 6 dataset configurations, each containing:
            - name (str): Dataset name for identification
            - dataroot_gt (str): Data root path for gt
            - dataroot_lq (str): Data root path for lq  
            - augment_factor (int): Multiplication factor for data augmentation
            - weight (float): Sampling weight for this dataset (optional)
            - Other standard dataset options (io_backend, filename_tmpl, etc.)
        batch_size (int): Batch size for training
        Other standard options: gt_size, use_hflip, use_rot, mean, std, etc.
    """
    
    def __init__(self, opt):
        super(MultiDatasetPairedImageDataset, self).__init__()
        self.opt = opt
        self.batch_size = opt.get('batch_size', 6)
        
        # Initialize 6 datasets
        self.datasets = []
        self.dataset_names = []
        self.augment_factors = []
        self.dataset_weights = []
        self.dataset_lengths = []
        # self.cumulative_lengths = [0]
        self.cumulative_lengths = [0]
        for L in self.dataset_lengths[:-1]:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + L)
                
        # Process each dataset configuration
        for i, dataset_opt in enumerate(opt['datasets']):
                
            dataset_info = self._init_single_dataset(dataset_opt)
            self.datasets.append(dataset_info)
            self.dataset_names.append(dataset_opt.get('name', f'dataset_{i}'))
            self.augment_factors.append(dataset_opt.get('augment_factor', 1))
            self.dataset_weights.append(dataset_opt.get('weight', 1.0))
            
            # Calculate dataset length with augmentation
            augmented_length = len(dataset_info['paths']) * self.augment_factors[i]
            self.dataset_lengths.append(augmented_length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + augmented_length)
        
        self.num_datasets = len(self.datasets)
        self.total_length = self.cumulative_lengths[-1]
        
        # Create sampling strategy for balanced batches
        self._create_batch_sampling_strategy()
        
        print(f"Initialized {self.num_datasets} datasets:")
        for i, name in enumerate(self.dataset_names):
            print(f"  {name}: {len(self.datasets[i]['paths'])} images "
                  f"-> {self.dataset_lengths[i]} (x{self.augment_factors[i]})")
    
    def _init_single_dataset(self, dataset_opt):
        """Initialize a single dataset configuration."""
        # File client setup
        io_backend_opt = dataset_opt['io_backend'].copy()
        
        gt_folder = dataset_opt['dataroot_gt']
        lq_folder = dataset_opt['dataroot_lq']
        filename_tmpl = dataset_opt.get('filename_tmpl', '{}')
        
        # Generate paths based on backend type
        if io_backend_opt['type'] == 'lmdb':
            io_backend_opt['db_paths'] = [lq_folder, gt_folder]
            io_backend_opt['client_keys'] = ['lq', 'gt']
            paths = paired_paths_from_lmdb([lq_folder, gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in dataset_opt and dataset_opt['meta_info_file'] is not None:
            paths = paired_paths_from_meta_info_file(
                [lq_folder, gt_folder], ['lq', 'gt'],
                dataset_opt['meta_info_file'], filename_tmpl)
        else:
            paths = paired_paths_from_folder([lq_folder, gt_folder], ['lq', 'gt'], filename_tmpl)
        
        return {
            'paths': paths,
            'io_backend_opt': io_backend_opt,
            'gt_folder': gt_folder,
            'lq_folder': lq_folder,
            'filename_tmpl': filename_tmpl
        }
    
    def _create_batch_sampling_strategy(self):
        """Create sampling strategy to ensure balanced representation in each batch."""
        # Calculate how many samples each dataset should contribute per batch
        base_samples_per_dataset = self.batch_size // self.num_datasets
        extra_samples = self.batch_size % self.num_datasets
        
        self.samples_per_dataset = [base_samples_per_dataset] * self.num_datasets
        
        # Distribute extra samples based on dataset weights
        if extra_samples > 0:
            # Sort datasets by weight (descending) and assign extra samples
            dataset_indices_by_weight = sorted(
                range(self.num_datasets), 
                key=lambda i: self.dataset_weights[i], 
                reverse=True
            )
            for i in range(extra_samples):
                dataset_idx = dataset_indices_by_weight[i % self.num_datasets]
                self.samples_per_dataset[dataset_idx] += 1
        
        print(f"Batch sampling strategy (batch_size={self.batch_size}):")
        for i, name in enumerate(self.dataset_names):
            print(f"  {name}: {self.samples_per_dataset[i]} samples per batch")
    
    def _get_dataset_and_local_index(self, global_index):
        """Convert global index to dataset index and local index within that dataset."""
        for dataset_idx in range(self.num_datasets):
            if global_index < self.cumulative_lengths[dataset_idx + 1]:
                local_index = global_index - self.cumulative_lengths[dataset_idx]
                return dataset_idx, local_index
        
        # Fallback to last dataset
        return self.num_datasets - 1, 0
    
    def _get_original_index(self, dataset_idx, local_index):
        """Get original image index considering augmentation factor."""
        original_dataset_length = len(self.datasets[dataset_idx]['paths'])
        original_index = local_index % original_dataset_length
        return original_index

    # ===== Add inside MultiDatasetPairedImageDataset =====
    def local_to_global_index(self, dataset_id: int, local_idx: int) -> int:
        """
        Map (dataset_id, local_idx) -> global flat index in [0, total_length).
        Assumes each sub-dataset occupies a consecutive segment:
            [cumulative_lengths[i], cumulative_lengths[i] + dataset_lengths[i])
        """
        local_len = int(self.dataset_lengths[dataset_id])
        li = int(local_idx) % local_len
        return int(self.cumulative_lengths[dataset_id] + li)

    def build_balanced_batch_sampler(self, batch_size, shuffle=True, drop_last=True, seed=42):
        """Factory to build a balanced sampler compatible with current pipeline."""
        return BalancedBatchSamplerV2(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )
    
    def __getitem__(self, index):
        # Determine which dataset and local index
        dataset_idx, local_index = self._get_dataset_and_local_index(index)
        original_index = self._get_original_index(dataset_idx, local_index)
        
        # Get dataset info
        dataset_info = self.datasets[dataset_idx]
        
        # Initialize file client if needed
        if not hasattr(self, f'file_client_{dataset_idx}') or getattr(self, f'file_client_{dataset_idx}') is None:
            file_client = FileClient(dataset_info['io_backend_opt'].pop('type'), **dataset_info['io_backend_opt'])
            setattr(self, f'file_client_{dataset_idx}', file_client)
        else:
            file_client = getattr(self, f'file_client_{dataset_idx}')
        
        scale = self.opt['scale']
        
        # Load gt and lq images
        gt_path = dataset_info['paths'][original_index]['gt_path']
        img_bytes = file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        
        lq_path = dataset_info['paths'][original_index]['lq_path']
        img_bytes = file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        
        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # Flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        
        # Color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
        
        # Crop unmatched GT images during validation/testing
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        # Normalize
        mean = self.opt.get('mean', None)
        std = self.opt.get('std', None)
        if mean is not None or std is not None:
            normalize(img_lq, mean, std, inplace=True)
            normalize(img_gt, mean, std, inplace=True)
        
        return {
            'lq': img_lq, 
            'gt': img_gt, 
            'lq_path': lq_path, 
            'gt_path': gt_path,
            'dataset_name': self.dataset_names[dataset_idx],
            'dataset_idx': dataset_idx
        }
    
    def __len__(self):
        return self.total_length


# class BalancedBatchSampler:
#     """Custom batch sampler to ensure balanced representation from all datasets."""
    
#     def __init__(self, dataset, batch_size, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_datasets = dataset.num_datasets
#         self.samples_per_dataset = dataset.samples_per_dataset
        
#         # Create indices for each dataset
#         self.dataset_indices = []
#         for i in range(self.num_datasets):
#             start_idx = dataset.cumulative_lengths[i]
#             end_idx = dataset.cumulative_lengths[i + 1]
#             indices = list(range(start_idx, end_idx))
#             if self.shuffle:
#                 random.shuffle(indices)
#             self.dataset_indices.append(indices)
        
#         # Calculate number of batches
#         min_batches_per_dataset = min(
#             len(indices) // samples for indices, samples 
#             in zip(self.dataset_indices, self.samples_per_dataset)
#             if samples > 0
#         )
#         self.num_batches = min_batches_per_dataset
    
#     def __iter__(self):
#         for batch_idx in range(self.num_batches):
#             batch = []
            
#             # Sample from each dataset according to the strategy
#             for dataset_idx in range(self.num_datasets):
#                 num_samples = self.samples_per_dataset[dataset_idx]
#                 start_pos = batch_idx * num_samples
#                 end_pos = start_pos + num_samples
                
#                 if end_pos <= len(self.dataset_indices[dataset_idx]):
#                     batch.extend(self.dataset_indices[dataset_idx][start_pos:end_pos])
            
#             if self.shuffle:
#                 random.shuffle(batch)
            
#             yield batch
    
#     def __len__(self):
#         return self.num_batches

# ===== Add to paired_image_dataset.py =====

class BalancedBatchSamplerV2(Sampler):
    """Yield *batches* of indices so each batch is as balanced as possible across sub-datasets.

    Rules:
      - If B = N*k: take k samples per dataset.
      - If B = N*k + r (0<r<N): take k per dataset, then add +1 to r distinct random datasets.
      - If B < N: take 1 sample from B distinct datasets (no duplicate).
    Notes:
      - Works for any N >= 1 (6, 9, 15, ...).
      - Shuffles per-epoch; restarts pools when a sub-dataset runs out.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=42):
        assert hasattr(dataset, 'num_datasets') and dataset.num_datasets >= 1, \
            'BalancedBatchSamplerV2 expects MultiDatasetPairedImageDataset with num_datasets >= 1'
        super().__init__()
        self.dataset = dataset
        self.N = int(dataset.num_datasets)
        self.B = int(batch_size)
        assert self.B >= 1, 'batch_size must be >= 1'
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0  # track current epoch

        # Per-dataset global index pools (shuffled, looped)
        self._rng = random.Random(self.seed)
        self.pools = []
        for did in range(self.N):
            local_len = int(dataset.dataset_lengths[did])
            local_indices = list(range(local_len))
            if self.shuffle:
                self._rng.shuffle(local_indices)
            # map local->global once to avoid重复计算
            global_indices = [dataset.local_to_global_index(did, li) for li in local_indices]
            self.pools.append({'data': global_indices, 'cursor': 0})

        # Epoch length（与现有统计口径对齐：总样本 / B 向上取整）
        self.num_batches = math.ceil(dataset.total_length / float(self.B))

        # 预生成每个 batch 的“多出的 r 份额”的数据集选择，保证可复现与平均
        self.base = self.B // self.N
        self.rem = self.B % self.N
        self.extra_alloc = []
        if self.rem > 0:
            for _ in range(self.num_batches):
                order = list(range(self.N))
                if self.shuffle:
                    self._rng.shuffle(order)
                self.extra_alloc.append(order[:self.rem])
        else:
            self.extra_alloc = [()] * self.num_batches

        self.distinct_select = (self.B < self.N)

    def __len__(self):
        if self.drop_last:
            full = (self.dataset.total_length // self.B)
            return max(1, full)  # 至少 1 个 batch（防止为 0）
        return self.num_batches

    def _pop(self, did, k, rng):
        pool = self.pools[did]
        out = []
        for _ in range(k):
            if pool['cursor'] >= len(pool['data']):
                # 重启并重洗
                data = list(pool['data'])
                if self.shuffle:
                    rng.shuffle(data)
                pool['data'] = data
                pool['cursor'] = 0
            out.append(pool['data'][pool['cursor']])
            pool['cursor'] += 1
        return out

    def set_epoch(self, epoch: int):
        """Called by training loop each epoch to reshuffle deterministically per-epoch."""
        self.epoch = int(epoch)

    def __iter__(self):
        # vary RNG with epoch to change order every epoch
        rng = random.Random(self.seed + self.epoch)
        for b in range(len(self)):
            batch = []
            if self.distinct_select:
                # 取 B 个互不相同的数据集，各 1 张
                order = list(range(self.N))
                if self.shuffle:
                    rng.shuffle(order)
                chosen = order[:self.B]
                for did in chosen:
                    batch += self._pop(did, 1, rng)
            else:
                # 每个数据集 base 张
                for did in range(self.N):
                    if self.base > 0:
                        batch += self._pop(did, self.base, rng)
                # r 个数据集各 +1 张
                for did in self.extra_alloc[b]:
                    batch += self._pop(did, 1, rng)

            if self.shuffle:
                rng.shuffle(batch)

            # drop_last 需要严格等于 B
            if self.drop_last and len(batch) != self.B:
                continue
            yield batch