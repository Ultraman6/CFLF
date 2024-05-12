import copy
from concurrent.futures import ThreadPoolExecutor

from algorithm.base.server import BaseServer
from model.base.model_dict import (_modeldict_weighted_average)


class RANK_API(BaseServer):
    def __init__(self, task):
        super().__init__(task)

    def local_update(self):
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户位次")
        cid_acc = {}
        if self.args.train_mode == 'serial':
            for cid in self.client_indexes:
                acc, _ = self.client_list[cid].local_test(valid=self.valid_global, mode='cooper')
                cid_acc[cid] = acc
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self.thread_test, cid=cid, w=None,
                                                valid=self.valid_global, origin=False, mode='cooper') for cid in
                           self.client_indexes}
                for cid, future in futures.items():
                    (acc, _), mes = future.result()
                    self.task.control.set_statue('text', mes)
                    cid_acc[cid] = acc
        sorted_cid = [cid for cid, acc in sorted(cid_acc.items(), key=lambda item: item[1])]
        for idx, cid in enumerate(sorted_cid):  # 位次应该等同数量
            w_locals = [copy.deepcopy(self.w_locals[sorted_cid[i]]) for i in range(idx + 1)]
            self.local_params[cid] = _modeldict_weighted_average(w_locals)
            # print(self.local_params[cid])
            self.task.control.set_info('local', 'position', (self.round_idx, idx + 1), cid)
        self.task.control.set_statue('text', f"结束计算客户位次")
        # super().local_update()  # 别忘了把奖励分配下去
