import concurrent
import copy
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from matplotlib import pyplot as plt
from scipy.stats import qmc, wasserstein_distance
import numpy as np
from tqdm import tqdm


class Sample_Generator:
    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

    def set_range(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

    def generate_samples_sobol(self, num_samples):
        """Generate more evenly distributed 2D samples using the Sobol sequence."""
        sampler = qmc.Sobol(d=2, scramble=True)
        samples = sampler.random(n=num_samples)
        samples_scaled = qmc.scale(samples, [self.x_range[0], self.y_range[0]],
                                   [self.x_range[1], self.y_range[1]])
        # 对x轴的值进行四舍五入并转换为整数
        samples_scaled[:, 0] = np.round(samples_scaled[:, 0]).astype(int)
        return samples_scaled

    def generate_samples_halton(self, num_samples):
        """Generate more evenly distributed 2D samples using the Halton sequence."""
        sampler = qmc.Halton(d=2, scramble=False)
        samples = sampler.random(n=num_samples)
        samples_scaled = qmc.scale(samples, [self.x_range[0], self.y_range[0]],
                                   [self.x_range[1], self.y_range[1]])
        # 对x轴的值进行四舍五入并转换为整数
        samples_scaled[:, 0] = np.round(samples_scaled[:, 0]).astype(int)
        return samples_scaled

    def generate_samples_stratified(self, num_samples):
        """Generate samples using stratified sampling."""
        num_layers = int(np.ceil(np.sqrt(num_samples)))
        adjusted_num_samples = num_layers ** 2
        x_layer_size = (self.x_range[1] - self.x_range[0]) / num_layers
        y_layer_size = (self.y_range[1] - self.y_range[0]) / num_layers

        samples = []
        for x_layer in range(num_layers):
            for y_layer in range(num_layers):
                x_low = self.x_range[0] + x_layer * x_layer_size
                x_high = x_low + x_layer_size
                y_low = self.y_range[0] + y_layer * y_layer_size
                y_high = y_low + y_layer_size

                sample = np.random.uniform(low=[x_low, y_low], high=[x_high, y_high], size=(1, 2))
                samples.append(sample)

        samples = np.vstack(samples)
        if adjusted_num_samples > num_samples:
            samples = samples[np.random.choice(samples.shape[0], num_samples, replace=False)]
        return samples

    def generate_samples_grid(self, num_samples):
        """Generate evenly spaced 2D distribution samples using grid sampling."""
        num_points_x = int(np.round(np.sqrt(num_samples)))
        num_points_y = int(np.round(np.sqrt(num_samples)))

        while num_points_x * num_points_y != num_samples:
            if num_points_x * num_points_y < num_samples:
                num_points_x += 1
            else:
                num_points_y -= 1

        # 确保x_values为整数
        x_values = np.linspace(self.x_range[0], self.x_range[1], num_points_x, endpoint=True)
        x_values = np.round(x_values).astype(int)  # 转换为整数并四舍五入
        y_values = np.linspace(self.y_range[0], self.y_range[1], num_points_y, endpoint=True)

        x, y = np.meshgrid(x_values, y_values)
        return np.vstack([x.ravel(), y.ravel()]).T

    @staticmethod
    def visualize_samples(samples, title="Sample Distribution"):
        """Visualize sample points."""
        x, y = samples[:, 0], samples[:, 1]
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title(title)
        plt.grid(True)
        plt.show()


class Dis_Generator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.uniform_dist = [1 / self.num_classes] * self.num_classes
        self.emd_range = self.get_emd_range()

    @staticmethod
    def is_finite_decimal(num):
        # Check if 1/num is a finite decimal in base 10
        while num % 2 == 0:
            num /= 2
        while num % 5 == 0:
            num /= 5
        return num == 1

    @staticmethod
    def visualize_errors(results):
        errors_emd = []
        errors_diff = []
        for re in results:
            errors_diff.append(re[0])
            errors_emd.append(re[1])

        plt.figure(figsize=(10, 6))
        plt.scatter(errors_diff, errors_emd, color='blue', label='EMD Error')
        plt.title('Error Visualization')
        plt.xlabel('Sample Error/Id')
        plt.ylabel('EMD Error (Absolute Difference)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_emd_range(self):
        milestones = self.calculate_milestones()
        return next(iter(milestones)), next(reversed(milestones))

    def calculate_uniform_distribution(self, num_classes):
        # Calculate a uniform distribution for num_classes
        if self.is_finite_decimal(num_classes):
            return [1 / num_classes] * num_classes + [0] * (self.num_classes - num_classes)
        return None

    def calculate_milestones(self):
        remaining_classes = list(range(self.num_classes))
        milestones = {}
        class_gap = []
        while remaining_classes:
            if self.is_finite_decimal(len(remaining_classes)):
                milestone_dist = [1 / len(remaining_classes) if i in remaining_classes else 0 for i in
                                  range(self.num_classes)]
                emd = self.calculate_emd(milestone_dist)
                milestones[emd] = (milestone_dist, copy.deepcopy(class_gap))
                class_gap.clear()
            if len(remaining_classes) == 1:
                break
            class_to_remove = np.random.choice(remaining_classes)
            remaining_classes.remove(class_to_remove)
            class_gap.append(class_to_remove)
        return milestones

    def calculate_emd(self, distribution):
        # Calculate the EMD for a given distribution against a uniform distribution
        emd = self.num_classes * wasserstein_distance(distribution, self.uniform_dist)
        return round(emd, 6)

    @staticmethod
    def find_nearest_milestones(emd, milestones):
        # Convert the dictionary keys to a list and sort them, just in case they aren't already
        sorted_emds = sorted(milestones.keys())
        lower_milestone_emd = None
        upper_milestone_emd = None
        for i in range(len(sorted_emds) - 1):
            if sorted_emds[i] <= emd <= sorted_emds[i + 1]:
                lower_milestone_emd = sorted_emds[i]
                upper_milestone_emd = sorted_emds[i + 1]
                break
        # If no exact match or range is found, handle edge cases
        if lower_milestone_emd is None and upper_milestone_emd is None:
            if emd < sorted_emds[0]:
                upper_milestone_emd = sorted_emds[0]  # emd is below the lowest milestone
            elif emd > sorted_emds[-1]:
                lower_milestone_emd = sorted_emds[-1]  # emd is above the highest milestone

        return lower_milestone_emd, upper_milestone_emd

    def gen_dis_from_emd(self, milestones, target_emd, max_iter=100000, tolerance=1e-6, up=0.00001, low=0, delta=1):
        # 寻找最接近的两个里程碑
        lower_emd, upper_emd = self.find_nearest_milestones(target_emd, milestones)
        current_distribution = np.array(milestones[upper_emd][0])
        current_emd = upper_emd
        if current_emd == target_emd:
            return current_distribution
        elif target_emd == lower_emd:
            return milestones[lower_emd][0]

        class_gap = milestones[upper_emd][1]
        for _ in range(max_iter):
            # 根据当前EMD与目标EMD的关系确定调整方向
            # direction = 1 if current_emd > target_emd else -1
            adjustment_index = np.random.choice(class_gap)
            adjustment_amount = np.random.uniform(low, up)  # 调整量大小和方向
            temp_distribution = current_distribution.copy()
            temp_distribution[adjustment_index] += adjustment_amount
            # 确保概率有效
            temp_distribution = np.clip(temp_distribution, 0, 1)
            temp_distribution /= temp_distribution.sum()  # 归一化

            # 计算调整后的EMD并判断是否接近目标
            new_emd = self.calculate_emd(temp_distribution)
            if abs(new_emd - target_emd) <= tolerance:
                return temp_distribution  # 尝试有效
            elif (new_emd > target_emd) and (new_emd < current_emd):
                # 如果调整使我们更接近目标，或者在正确的方向上移动，则接受调整
                current_distribution = temp_distribution
                current_emd = new_emd
                up *= delta

        return current_distribution

    def parse_sample_process(self, sample):
        sample_size, target_emd = sample
        milestones = self.calculate_milestones()
        distribution = self.gen_dis_from_emd(milestones, target_emd)
        generated_emd = self.calculate_emd(distribution)
        error = abs(target_emd - generated_emd)
        return sample_size, distribution, generated_emd, error

    def parse_samples(self, samples):
        with Pool(processes=5) as pool:
            results = list(tqdm(pool.imap(self.parse_sample_process, samples), total=len(samples)))
        return results

    def parse_real_dis_process(self, result, num_bar=7, emd_tolerance=1e-3):
        sample_size, distribution, target_emd, _ = result  # 暂时无法解决取整带来的误差
        # print(f"sample size: {sample_size}" + f"dis: {distribution}" + f"emd: {target_emd}")
        # 计算初始真实分布
        real_distribution = np.round(sample_size * np.array(distribution)).astype(int)
        original_distribution = np.array(distribution)
        # 记录初始时为0的元素索引
        zero_indices = np.where(original_distribution == 0)[0]
        diff = sample_size - np.sum(real_distribution)
        new_emd = self.calculate_emd(real_distribution / np.sum(real_distribution))
        emd_error = target_emd - new_emd
        # if abs(diff) > num_bar and emd_error > emd_tolerance:
        # while abs(diff) > num_bar and abs(emd_error) > emd_tolerance:
        #     # 计算分布差异 将初始为0的元素的差异设置为无效值，以便排除它们
        #     distribution_diff = real_distribution / np.sum(real_distribution) - original_distribution
        #     distribution_diff[zero_indices] = np.inf if diff > 0 else -np.inf
        #     if diff > 0:
        #         idx_to_increase = np.argmin(distribution_diff)
        #         real_distribution[idx_to_increase] += 1
        #         diff -= 1
        #     elif diff < 0:
        #         idx_to_decrease = np.argmax(distribution_diff)
        #         real_distribution[idx_to_decrease] -= 1
        #         diff += 1
        #     new_emd = self.calculate_emd(real_distribution / np.sum(real_distribution))
        #     emd_error = target_emd - new_emd
        # if emd_error < emd_tolerance:
        #     break

        sample_size -= diff
        return sample_size, real_distribution, new_emd, diff, emd_error
    def parse_real_dis(self, results):
        with Pool(processes=5) as pool:
            results_new = list(tqdm(pool.imap(self.parse_real_dis_process, results), total=len(results)))
        return results_new


class Data_Generator:
    def __init__(self, num_classes=10, samples_range=(500, 5000)):
        self.dis_generator = Dis_Generator(num_classes)
        self.sam_generator = Sample_Generator(samples_range, self.dis_generator.emd_range)

    def get_real_samples(self, num_samples, mode='grid', visual=True):
        if mode == 'grid':
            samples_grid = self.sam_generator.generate_samples_grid(num_samples)
        elif mode == 'sobol':
            samples_grid = self.sam_generator.generate_samples_sobol(num_samples)
        elif mode == 'halton':
            samples_grid = self.sam_generator.generate_samples_halton(num_samples)
        else:
            raise ValueError("Unsupported mode")

        results = self.dis_generator.parse_samples(samples_grid)
        results_new = self.dis_generator.parse_real_dis(results)
        samples_real = np.array([(s, e) for s, _, e, _, _ in results_new])

        if visual:
            self.sam_generator.visualize_samples(samples_grid, mode + " sampling")
            error_samples = [(i, e) for i, (_, _, _, e) in enumerate(results)]
            self.dis_generator.visualize_errors(error_samples)
            self.sam_generator.visualize_samples(samples_real, mode + " real sampling")
            error_real = [(e1, e2) for _, _, _, e1, e2 in results_new]
            self.dis_generator.visualize_errors(error_real)

            # 计算两组样本之间的相关性系数
            des_samples = np.array(samples_grid)
            real_samples = np.array([(e1, e2) for e1, _, e2, _, _ in results_new])
            correlation_x = np.corrcoef(des_samples[:, 0], real_samples[:, 0])[0, 1]
            correlation_y = np.corrcoef(des_samples[:, 1], real_samples[:, 1])[0, 1]
            print(correlation_x, correlation_y)

        results_final = [(s, d, e) for s, d, e, _, _ in results_new]
        return results_final


if __name__ == '__main__':
    generator = Data_Generator(1000, (10000, 50000))
    samples = generator.get_real_samples(1000, 'grid', True)
