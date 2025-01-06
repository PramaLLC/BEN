import os
import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils import data
import cv2







class val_dataset(data.Dataset):
    def __init__(self, image_root, gt_root):
        self.img_list_1 = sorted([os.path.splitext(f)[0] for f in os.listdir(image_root)
                                  if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')])
        self.img_list_2 = sorted([os.path.splitext(f)[0] for f in os.listdir(gt_root)
                                  if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')])

        # Intersection of all lists to get common images
        self.img_list = sorted(list(set(self.img_list_1) & set(self.img_list_2)))

        print("SELF IMAGE LIST: ", len(self.img_list))
        self.image_root = image_root
        self.gt_root = gt_root

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        self.size = len(self.img_list)
        print("Dataset size: ", self.size)

    def __getitem__(self, index):
        # Ensure the index is within bounds
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")

        base_name = self.img_list[index]
        rgb_paths = [
            os.path.join(self.image_root, base_name + ext) for ext in ['.png', '.jpg', '.bmp']
        ]
        gt_paths = [
            os.path.join(self.gt_root, base_name + ext) for ext in ['.png', '.jpg', '.bmp']
        ]


        # Find the existing image path
        for path in rgb_paths:
            if os.path.exists(path):
                image = self.binary_loader(path)
                break
        else:
            raise FileNotFoundError(f"Image file for {base_name} not found in {self.image_root}")

        # Find the existing ground truth path
        for path in gt_paths:
            if os.path.exists(path):
                gt = self.binary_loader(path)
                break
        else:
            raise FileNotFoundError(f"GT file for {base_name} not found in {self.gt_root}")


        # Convert images to NumPy arrays
        image = np.array(image)
        gt = np.array(gt)


        # Resize images to (1024, 1024)
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        return image, gt

    def load_data(self, index):
        """Custom method to load data for a given index."""
        return self.__getitem__(index)



    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


    def __len__(self):
        return self.size
    


def format_metric(name, value):
    def format_value(v):
        if isinstance(v, (int, float, np.number)):
            return f'{v:.4f}'
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                return f'{v.item():.4f}'
            else:
                return '_'.join(format_value(x) for x in v.flatten())
        elif isinstance(v, (list, tuple)):
            return '_'.join(format_value(x) for x in v)
        else:
            return str(v)

    formatted_value = format_value(value)
    return f"{name}_{formatted_value}"



class MaxFMeasure:
    def __init__(self):
        self.beta2 = 0.3  # As per the paper
        self.precisions = []
        self.recalls = []
        self.fmeasures = []
        
    def update(self, pred, gt):
        # Convert inputs to numpy if they're not already
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
            
        # Ensure inputs are numpy arrays and proper dimensions
        pred = pred.squeeze()
        gt = gt.squeeze()
        
        # Thresholds for F-measure evaluation
        thresholds = np.linspace(0, 1, 256)
        
        prec, rec, fms = [], [], []
        for thresh in thresholds:
            binary_pred = (pred >= thresh).astype(np.float32)
            binary_gt = (gt > 0.5).astype(np.float32)  # Ensure GT is binary
            
            # Calculate precision and recall
            tp = (binary_pred * binary_gt).sum()
            fp = binary_pred.sum() - tp
            fn = binary_gt.sum() - tp
            
            prec_t = tp / (tp + fp + 1e-8)
            rec_t = tp / (tp + fn + 1e-8)
            
            # Calculate F-measure
            fm_t = (1 + self.beta2) * prec_t * rec_t / (self.beta2 * prec_t + rec_t + 1e-8)
            
            prec.append(prec_t)
            rec.append(rec_t)
            fms.append(fm_t)
            
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.fmeasures.append(fms)
    
    def show(self):
        if not self.fmeasures:
            return 0.0, 0.0
        all_fms = np.array(self.fmeasures)
        max_f = all_fms.max(axis=1).mean()  # MaxFm
        mean_f = all_fms.mean(axis=1).mean()  # MeanFm
        return max_f, mean_f
class WeightedFMeasure:
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        # Convert inputs to numpy if they're not already
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
            
        # Ensure proper dimensions
        pred = pred.squeeze()
        gt = gt.squeeze()
        
        # Input validation
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        # Convert GT to binary
        gt = gt > 0.5

        if gt.max() == 0:  # Empty GT
            score = 0
        else:
            score = self._calculate_score(pred, gt)
            
        self.scores_list.append(score)

    def _calculate_score(self, pred, gt):
        # Calculate distance transform and indices
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # Calculate error and apply edge preservation
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # Apply Gaussian filtering
        K = self._gaussian_kernel(shape=(7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        
        # Get minimum error
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # Calculate pixel importance
        B = np.where(gt == 0, 
                    2 - np.exp(np.log(0.5) / 5 * Dst), 
                    np.ones_like(gt))
        Ew = MIN_E_EA * B

        # Calculate weighted precision and recall
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        R = 1 - np.mean(Ew[gt])
        P = TPw / (self.eps + TPw + FPw)

        # Calculate weighted F-measure
        Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return Q

    def _gaussian_kernel(self, shape=(7, 7), sigma=5):
        """
        Generate 2D Gaussian kernel
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        
        # Remove small values
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        
        # Normalize
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
            
        return h

    def show(self):
        return np.mean(self.scores_list) if self.scores_list else 0
    

class EMeasure:
    def __init__(self):
        self.scores = []

    def update(self, pred, gt):
        # Convert inputs to numpy if they're not already
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
            
        # Ensure proper dimensions
        pred = pred.squeeze()
        gt = gt.squeeze()
        
        # Threshold ground truth
        gt = (gt > 0.5).astype(np.float32)
        
        # Adaptive threshold for prediction
        thresh = 2 * pred.mean()
        thresh = min(thresh, 1.0)
        binary_pred = (pred >= thresh).astype(np.float32)
        
        if np.sum(gt) == 0:  # Empty ground truth
            score = 1 - binary_pred.mean()
        elif np.sum(1 - gt) == 0:  # Full ground truth
            score = binary_pred.mean()
        else:
            # Alignment matrix
            align_matrix = self._alignment_term(binary_pred, gt)
            enhanced_align_matrix = self._enhanced_alignment(align_matrix)
            score = enhanced_align_matrix.mean()
        
        self.scores.append(score)

    def _alignment_term(self, pred, gt):
        mu_pred = pred.mean()
        mu_gt = gt.mean()
        
        align_pred = pred - mu_pred
        align_gt = gt - mu_gt
        
        align_matrix = 2 * (align_gt * align_pred) / (align_gt**2 + align_pred**2 + 1e-8)
        return align_matrix

    def _enhanced_alignment(self, align_matrix):
        return (align_matrix + 1)**2 / 4

    def show(self):
        return np.mean(self.scores) if self.scores else 0.0
class SMeasure:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.scores = []

    def update(self, pred, gt):
        # Convert inputs to numpy if they're not already
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
            
        # Ensure proper dimensions
        pred = pred.squeeze()
        gt = gt.squeeze()
        
        # Convert ground truth to binary
        gt = (gt > 0.5).astype(np.float32)
        
        if gt.sum() == 0:  # Empty ground truth
            score = 1 - pred.mean()
        else:
            score = self.alpha * self._object_score(pred, gt) + \
                    (1 - self.alpha) * self._region_score(pred, gt)
        
        self.scores.append(score)

    def _object_score(self, pred, gt):
        # Object-aware Component
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        
        u = np.mean(gt)
        
        # Object-aware Component
        o_fg = self._s_object(fg, gt)
        o_bg = self._s_object(bg, 1 - gt)
        
        return u * o_fg + (1 - u) * o_bg

    def _s_object(self, region, gt):
        x = np.mean(region[gt > 0])
        sigma_x = np.std(region[gt > 0])
        return 2 * x / (x**2 + 1 + sigma_x + 1e-8)

    def _region_score(self, pred, gt):
        # Region-aware Component
        y, x = ndimage.center_of_mass(gt)
        y, x = int(y), int(x)
        
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divide_regions(gt, y, x)
        p1, p2, p3, p4 = self._divide_regions(pred, y, x)[:4]
        
        # Calculate region similarity
        s1 = self._ssim(p1, gt1)
        s2 = self._ssim(p2, gt2)
        s3 = self._ssim(p3, gt3)
        s4 = self._ssim(p4, gt4)
        
        return s1*w1 + s2*w2 + s3*w3 + s4*w4

    def _ssim(self, pred, gt):
        # Simplified SSIM calculation
        c1, c2 = 0.01**2, 0.03**2
        mu_x = np.mean(pred)
        mu_y = np.mean(gt)
        sigma_x = np.std(pred)
        sigma_y = np.std(gt)
        sigma_xy = np.mean((pred - mu_x) * (gt - mu_y))
        
        ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denom = (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
        return ssim / (denom + 1e-8)

    def _divide_regions(self, img, y, x):
        h, w = img.shape
        area = h * w
        
        lt = img[:y, :x]
        rt = img[:y, x:]
        lb = img[y:, :x]
        rb = img[y:, x:]
        
        # Weights
        w1 = (y * x) / area
        w2 = (y * (w - x)) / area
        w3 = ((h - y) * x) / area
        w4 = ((h - y) * (w - x)) / area
        
        return lt, rt, lb, rb, w1, w2, w3, w4

    def show(self):
        return np.mean(self.scores)


    
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist

class cal_mae(object):
    # Mean Absolute Error
    def __init__(self):
        self.sum_mae = 0.0
        self.count = 0
        self.total = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.sum_mae += score
        self.count += 1

    def cal(self, pred, gt):
        score = np.mean(np.abs(pred - gt))
        self.total.append(score)
        return score

    def show(self):
        return self.sum_mae / self.count if self.count != 0 else 0

class cal_dice(object):
    # Dice Coefficient
    def __init__(self):
        self.sum_dice = 0.0
        self.count = 0

    def update(self, y_pred, y_true):
        score = self.cal(y_pred, y_true)
        self.sum_dice += score
        self.count += 1

    def cal(self, y_pred, y_true):
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def show(self):
        return self.sum_dice / self.count if self.count != 0 else 0

class cal_iou(object):
    # Intersection over Union
    def __init__(self):
        self.sum_iou = 0.0
        self.count = 0

    def update(self, input, target):
        score = self.cal(input, target)
        self.sum_iou += score
        self.count += 1

    def cal(self, input, target):
        smooth = 1e-5
        input_bin = input > 0.5
        target_bin = target > 0.5
        intersection = np.logical_and(input_bin, target_bin).sum()
        union = np.logical_or(input_bin, target_bin).sum()
        return (intersection + smooth) / (union + smooth)

    def show(self):
        return self.sum_iou / self.count if self.count != 0 else 0

class cal_ber(object):
    # Balanced Error Rate
    def __init__(self):
        self.sum_ber = 0.0
        self.count = 0

    def update(self, y_pred, y_true):
        score = self.cal(y_pred, y_true)
        self.sum_ber += score
        self.count += 1

    def cal(self, y_pred, y_true):
        binary = y_pred >= 0.5
        hard_gt = y_true > 0.5
        tp = np.logical_and(binary, hard_gt).sum()
        tn = np.logical_and(~binary, ~hard_gt).sum()
        Np = hard_gt.sum()
        Nn = (~hard_gt).sum()
        ber = 1 - ((tp / (Np + 1e-8) + tn / (Nn + 1e-8)) / 2)
        return ber

    def show(self):
        return self.sum_ber / self.count if self.count != 0 else 0

class cal_acc(object):
    # Accuracy
    def __init__(self):
        self.sum_acc = 0.0
        self.count = 0

    def update(self, y_pred, y_true):
        score = self.cal(y_pred, y_true)
        self.sum_acc += score
        self.count += 1

    def cal(self, y_pred, y_true):
        binary = y_pred >= 0.5
        hard_gt = y_true > 0.5
        correct = np.equal(binary, hard_gt).sum()
        total = binary.size
        acc = correct / total
        return acc

    def show(self):
        return self.sum_acc / self.count if self.count != 0 else 0






# Define the image transformation
img_transform = transforms.Compose([
    transforms.ToTensor(),    
])




def test_model( test_loader):
    try:
        num_images =  test_loader.size
    
        mae = cal_mae()
        m_dice = cal_dice()
        m_iou = cal_iou()
        ber = cal_ber()
        acc = cal_acc()
        max_fm = MaxFMeasure()
        s_measure = SMeasure()
        e_measure = EMeasure()
        wfm = WeightedFMeasure()

        for index in range(num_images):
            
            print("START")
            res, gt = test_loader.load_data(index)
            print("HERER")

            res_tensor = img_transform(res).unsqueeze(0)
            gt_tensor = img_transform(gt).unsqueeze(0)
            res = res_tensor.detach().cpu().numpy()
            gt_np = gt_tensor.detach().cpu().numpy()




            mae.update(res, gt_np)
            m_dice.update(res, gt_np)
            m_iou.update(res, gt_np)
            ber.update(res, gt_np)
            acc.update(res, gt_np)
            max_fm.update(res, gt_np)
            s_measure.update(res, gt_np)
            e_measure.update(res, gt_np)
            wfm.update(res, gt_np)


        
        
        

        # Get metrics at the end
        MAE = mae.show()
        m_dice_value = m_dice.show()
        m_iou_value = m_iou.show()
        ber_value = ber.show()
        acc_value = acc.show()
        max_f, mean_f = max_fm.show()
        s_score = s_measure.show()
        e_score = e_measure.show()
        wfm = wfm.show()
        metrics = [
            ("MAE", MAE),
            ("DICE", m_dice_value),
            ("IOU", m_iou_value),
            ("BER", ber_value),
            ("ACC", acc_value),
            ("MaxF", max_f),
            ("MeanF", mean_f),
            ("S-measure", s_score),
            ("E-measure", e_score),
            ("weighted f", wfm)
        ]


        formatted_metrics = ''.join([format_metric(name, value) for name, value in metrics])
        max_length = 255
        if len(formatted_metrics) > max_length:
            formatted_metrics = formatted_metrics[:max_length - 3] + '...'
        filepath = f"results{formatted_metrics}"
        print("testedModel: ", filepath)
        print("MAE IS ", MAE)
        return filepath, metrics

    except Exception as e:
        print(f"Error in test_model: {e}")
        return None, None









def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seed(8)



gt_root = 'BEN_validation/gt'
prediction_root = "BEN_validation/ben_base+refiner"



evalution_dataset = val_dataset(prediction_root,  gt_root)
score = test_model(evalution_dataset)
