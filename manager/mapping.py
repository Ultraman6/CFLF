from algorithm.base.server import BaseServer
from algorithm.method.auto_fusion.auto_fusion import Auto_Fusion_API
from algorithm.method.auto_fusion.auto_fusion_layer import Auto_Fusion_Layer_API
from algorithm.method.cosine_similarity_reward.common import CS_Reward_API
from algorithm.method.cosine_similarity_reward.just_outer import CS_Reward_Out_API
from algorithm.method.cosine_similarity_reward.reputation import CS_Reward_Reputation_API
from algorithm.method.dot_attention.dot_layer_att import Layer_Att_API
from algorithm.method.dot_attention.layer_att_cross_up import Cross_Up_Att_API
from algorithm.method.dot_quality.margin_dot import Margin_Dot_API
from algorithm.method.gradient_influence.fedavg_api import Grad_Inf_API
from algorithm.method.gradnorm_ood.gradnorm import Grad_Norm_API
from algorithm.method.margin_Info.cross_info import Margin_Cross_Info_API
from algorithm.method.margin_JSD.common import Margin_JSD_Common_API
from algorithm.method.margin_JSD.direct_sum import Margin_JSD_Direct_Sum_API
from algorithm.method.margin_KL.div import Div_API
from algorithm.method.margin_KL.div_exp import Div_Exp_API
from algorithm.method.margin_KL.exp_div import Exp_Div_API
from algorithm.method.margin_KL.exp_sub_exp import Exp_Sub_Exp_API
from algorithm.method.margin_KL.sub_exp import Sub_Exp_API
from algorithm.method.margin_KL.sub_exp_exp import Sub_Exp_Exp_API
from algorithm.method.margin_KL.sub_exp_num import Sub_Exp_Num_API
from algorithm.method.margin_Loss.fedavg_api import MarginLossAPI
from algorithm.method.margin_Loss.gradnorm_update import Grad_Norm_Update_API
from algorithm.method.stage_two.fusion_mask import Fusion_Mask_API
from algorithm.method.up_metric.KL_update import JSD_Up_API
from algorithm.method.up_metric.cross_up_select import Cross_Up_Select_API
from algorithm.method.up_metric.cross_update_num import Cross_Up_Num_API
from algorithm.method.up_metric.loss_update import Loss_Up_API
from algorithm.method.up_metric.cross_update import Cross_Up_API
from algorithm.method.stage_two.margin_kl_cos_reward import Stage_Two_API
from algorithm.method.update_cluster.gradient_cluster import Up_Cluster_API

name_mapping = {
    'method': '方法', 'integrity': '完整算法'
}
algorithm_mapping = {
    'fedavg': BaseServer,
    'Margin_Loss': MarginLossAPI,
    'margin_dot': Margin_Dot_API,
    'loss_up': Loss_Up_API,
    'cross_up': Cross_Up_API,
    'cross_up_select': Cross_Up_Select_API,
    'cross_up_num': Cross_Up_Num_API,
    'up_cluster': Up_Cluster_API,
    'JSD_up': JSD_Up_API,
    'grad_norm_up': Grad_Norm_Update_API,
    'Margin_GradNorm': Grad_Norm_API,
    'MarginKL_sub_exp_exp': Sub_Exp_Exp_API,
    'MarginKL_sub_exp': Sub_Exp_API,
    'MarginKL_sub_exp_num': Sub_Exp_Num_API,
    'MarginKL_exp_sub_exp': Exp_Sub_Exp_API,
    'MarginKL_div': Div_API,
    'MarginKL_div_exp': Div_Exp_API,
    'MarginKL_exp_div': Exp_Div_API,
    'MarginJSD': Margin_JSD_Common_API,
    'MarginJSD_direct_sum': Margin_JSD_Direct_Sum_API,
    'MarginLoss_Cross_Info': Margin_Cross_Info_API,
    'layer_att': Layer_Att_API,
    'cross_up_att': Cross_Up_Att_API,
    'grad_inf': Grad_Inf_API,
    'auto_fusion': Auto_Fusion_API,
    'auto_layer_fusion': Auto_Fusion_Layer_API,
    'fusion_mask': Fusion_Mask_API,
    'Stage_two': Stage_Two_API,
    'Cosine_Similiarity_Reward': CS_Reward_API,
    'Cosine_Similarity_Out_Reward': CS_Reward_Out_API,
    'CS_Reward_Reputation': CS_Reward_Reputation_API,

}

algorithm_dict = {
    'method': {
        'aggregrate': {
            'Margin_Loss': MarginLossAPI,
            'margin_dot': Margin_Dot_API,
            'loss_up': Loss_Up_API,
            'cross_up': Cross_Up_API,
            'cross_up_select': Cross_Up_Select_API,
            'cross_up_num': Cross_Up_Num_API,
            'up_cluster': Up_Cluster_API,
            'JSD_up': JSD_Up_API,
            'grad_norm_up': Grad_Norm_Update_API,
            'Margin_GradNorm': Grad_Norm_API,
            'MarginKL_sub_exp_exp': Sub_Exp_Exp_API,
            'MarginKL_sub_exp': Sub_Exp_API,
            'MarginKL_sub_exp_num': Sub_Exp_Num_API,
            'MarginKL_exp_sub_exp': Exp_Sub_Exp_API,
            'MarginKL_div': Div_API,
            'MarginKL_div_exp': Div_Exp_API,
            'MarginKL_exp_div': Exp_Div_API,
            'MarginJSD': Margin_JSD_Common_API,
            'MarginJSD_direct_sum': Margin_JSD_Direct_Sum_API,
            'MarginLoss_Cross_Info': Margin_Cross_Info_API,

        },
        'fusion': {
            'layer_att': Layer_Att_API,
            'cross_up_att': Cross_Up_Att_API,
            'grad_inf': Grad_Inf_API,
            'auto_fusion': Auto_Fusion_API,
            'auto_layer_fusion': Auto_Fusion_Layer_API,
        },
        'reawrd': {
            'fusion_mask': Fusion_Mask_API,
            'Stage_two': Stage_Two_API,
            'Cosine_Similiarity_Reward': CS_Reward_API,
            'Cosine_Similarity_Out_Reward': CS_Reward_Out_API,
            'CS_Reward_Reputation': CS_Reward_Reputation_API,
        }
    },
    'integrity': {
        'base': {
            'fedavg': BaseServer,
        },
        'fair': {
            'qfll': BaseServer,
        }
    }
}
# 所有算法需要记录的信息全部在此进行形式化定义
# 参数分为三种类型：全局、局部、部分局部
# 不同算法可重新定义其信息结构，在common上加入即可
algo_info_mapping = {
    'common': {
        'global_info': {
            'Loss': '损失',
            'Accuracy': '精度',
            'Time': '时间戳',
        },
        'client_info': {
            'avg_loss': '平均损失',
            'learning_rate': '学习率',
        }
    },
    'fedetf': {

    }
}

