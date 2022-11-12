import conf
import random
import copy
import torch
import torch.nn.functional as F
import numpy as np

class FIFO():
    def __init__(self, capacity):
        self.data = [[], [], []]
        self.capacity = capacity
        pass

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

class Reservoir(FIFO): # Time uniform, motivated by CBRS

    def __init__(self, capacity):
        super(Reservoir, self).__init__(capacity)
        self.counter = 0


    def add_instance(self, instance):
        assert (len(instance) == 3)
        is_add = True
        self.counter+=1

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance()

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])


    def remove_instance(self):


        m = self.get_occupancy()
        n = self.counter
        u = random.uniform(0, 1)
        if u <= m / n:
            tgt_idx = random.randrange(0, m)  # target index to remove
            for dim in self.data:
                dim.pop(tgt_idx)
        else:
            return False
        return True

def get_diversity(logits):
    # shape: (batch_size, num_class)
    epsilon = 1e-6
    with torch.no_grad():
        soft_prob = F.softmax(logits, dim=1)
        pb_pred_tgt = soft_prob.mean(dim=0)
        target_div_loss = -torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + epsilon)))
        return target_div_loss

class Diversity(FIFO): # Maximize diversity

    def __init__(self, capacity):
        super(Diversity, self).__init__(capacity)
        self.data = [[], [], [], []]


    def add_instance_with_logit(self, instance, logit):
        instance = list(instance)
        instance.append(logit)
        assert (len(instance) == 4)
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance_with_input(instance)

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])

    def get_memory(self):
        return self.data[:3]

    def remove_instance_with_input(self, instance):

        min_diversity = get_diversity(torch.stack(self.data[3]).squeeze(1)) # current diversity
        pop_idx = -1

        for d_i in range(len(self.data[3])):
            logits = self.data[3][:d_i] + self.data[3][d_i+1:]
            logits.append(instance[3])
            diversity = get_diversity(torch.stack(logits))

            if diversity < min_diversity:
                min_diversity = diversity
                pop_idx = d_i

        if pop_idx >= 0 :
            for dim in self.data:
                dim.pop(pop_idx)
        else:
            return False
        return True

class CBRS():  # "Online Continual Learning from Imbalanced Data"

    def __init__(self, capacity):
        self.data = [[[], [], []] for _ in range(conf.args.opt['num_class'])]
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())

    def get_memory(self):

        if self.get_occupancy() < self.capacity:
            data = self.weighted_replay()
        else:
            data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def weighted_replay(self):
        replayed_data = copy.deepcopy(self.data)

        # calc probabilities
        occupancy_per_class = self.get_occupancy_per_class()
        prob_per_class = [1 / x if x > 0 else 0 for x in occupancy_per_class]
        scaling_factor = sum(prob_per_class)
        prob_per_class = [p / scaling_factor for p in prob_per_class]
        # print(prob_per_class)
        assert (sum(prob_per_class) - 1 < 0.01)

        is_full = False
        while not is_full:

            # replay random data
            tgt_cls = np.random.choice(np.arange(0, conf.args.opt['num_class']), p=prob_per_class)
            tgt_idx = random.randrange(0, len(self.data[tgt_cls][0]))  # target index to remove
            instance = self.data[tgt_cls][0][tgt_idx], self.data[tgt_cls][1][tgt_idx], self.data[tgt_cls][2][tgt_idx],
            for i, dim in enumerate(replayed_data[tgt_cls]):
                dim.append(instance[i])

            # calc size
            occupancy = 0
            for data_per_cls in replayed_data:
                occupancy += len(data_per_cls[0])
            if occupancy >= self.capacity:
                is_full = True
            else:
                is_full = False
        return replayed_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

        self.mark()

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def mark(self):
        for largest in self.get_largest_indices():
            self.marker[largest] = 'full'

    def remove_instance(self, cls):
        if self.marker[cls] != 'full': #  instance is stored in the place of another instance that belongs to the largest class
            largest_indices = self.get_largest_indices()
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True

class CBRS_debug():  # Store original label for debugging

    def __init__(self, capacity):
        self.data = [[[], [], [], []] for _ in range(conf.args.opt['num_class'])]
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        if self.get_occupancy() < self.capacity:
            data = self.weighted_replay()
        else:
            data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def weighted_replay(self):
        replayed_data = copy.deepcopy(self.data)

        # calc probabilities
        occupancy_per_class = self.get_occupancy_per_class()
        prob_per_class = [1 / x if x > 0 else 0 for x in occupancy_per_class]
        scaling_factor = sum(prob_per_class)
        prob_per_class = [p / scaling_factor for p in prob_per_class]
        # print(prob_per_class)
        assert (sum(prob_per_class) - 1 < 0.01)

        is_full = False
        while not is_full:

            # replay random data
            tgt_cls = np.random.choice(np.arange(0, conf.args.opt['num_class']), p=prob_per_class)
            tgt_idx = random.randrange(0, len(self.data[tgt_cls][0]))  # target index to remove
            instance = self.data[tgt_cls][0][tgt_idx], self.data[tgt_cls][1][tgt_idx], self.data[tgt_cls][2][tgt_idx],
            for i, dim in enumerate(replayed_data[tgt_cls][:3]):
                dim.append(instance[i])

            # calc size
            occupancy = 0
            for data_per_cls in replayed_data:
                occupancy += len(data_per_cls[0])
            if occupancy >= self.capacity:
                is_full = True
            else:
                is_full = False
        return replayed_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class


    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

        self.mark()

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def mark(self):
        for largest in self.get_largest_indices():
            self.marker[largest] = 'full'

    def remove_instance(self, cls):
        if self.marker[cls] != 'full': #  instance is stored in the place of another instance that belongs to the largest class
            largest_indices = self.get_largest_indices()
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True



class CBFIFO_debug():  # Store original label for debugging

    def __init__(self, capacity):
        self.data = [[[], [], [], [], []] for _ in range(conf.args.opt['num_class'])] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * conf.args.opt['num_class']
        self.marker = [''] * conf.args.opt['num_class']
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls, _, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * conf.args.opt['num_class']
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 5)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = 0
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            tgt_idx = 0
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

class CBReservoir_debug(CBFIFO_debug):  # Store original label for debugging

    def __init__(self, capacity):

        super(CBReservoir_debug, self).__init__(capacity)

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True