import copy
import json
import os
import traceback
from datetime import datetime
from sqlite3 import IntegrityError
from typing import Dict
from ex4nicegui import to_ref
from passlib.hash import bcrypt
from tortoise import fields, models
from tortoise.transactions import in_transaction
from visual.parts.constant import profile_dict, path_dict, ai_config_dict
from visual.parts.func import to_base64, get_image_data, move_all_files, convert_keys_to_int



class Experiment(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    time = fields.DatetimeField(auto_now_add=True)
    user = fields.ForeignKeyField('models.User', related_name='experiments')  # 正确的外键定义方式
    config = fields.JSONField()
    distribution = fields.JSONField(null=True)
    task_names = fields.JSONField(null=True)
    results = fields.JSONField(null=True)
    description = fields.TextField(null=True)

    @classmethod
    async def create_new_exp(cls, name: str, user: int, config: dict, dis: dict, task_names: dict, res: dict,
                             des: str) -> (bool, str):
        try:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_instance = await User.get(id=user)
            await cls.create(name=name, user=user_instance, time=time, config=config,
                             distribution=json.dumps(dis), task_names=json.dumps(task_names), results=json.dumps(res), description=des)
            return True, f"{name}信息保存成功"
        except IntegrityError:
            # 如果违反了数据库的唯一性约束等
            return False, "保存失败(完整性错误)"
        except Exception as e:
            # 捕获其他可能的异常，并返回错误消息
            traceback_details = traceback.format_exc()
            print(f"保存失败(其他错误): {traceback_details}")
            return False, f"保存失败(其他错误): {str(e)}"

    async def remove(self):
        try:
            await self.delete()
            return True, f"{self.name}数据删除成功"
        except IntegrityError:
            return False, "删除失败(完整性错误)"
        except Exception as e:
            return False, f"删除失败(其他错误): {str(e)}"


class User(models.Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=255, unique=True)  # 用户名作为登录凭据
    password = fields.CharField(max_length=255)
    profile = fields.JSONField(null=True)  # 包含学历和研究方向的JSON字段
    local_path = fields.JSONField(null=True)  # 包含三种本地路径的JSON字段
    avatar = fields.TextField(null=True)  # 用户头像base64字符串
    ai_config = fields.JSONField(null=True)
    share = fields.JSONField(null=True)   # 用户分享（id集合）
    shared = fields.JSONField(null=True)  # 用户接收到的分享（id集合）

    @staticmethod
    def get_tem_dict():
        profile = {}
        for k, v in profile_dict.items():
            v['default'] = to_ref('0')
            profile[k] = v
        path = {}
        for k, v in path_dict.items():
            v['default'] = to_ref(os.path.abspath(os.path.join('..', 'files', k)))
            v['name'] = v['name'] + '保存路径'
            path[k] = v

        return {
            'uname': to_ref(''),
            'pwd': to_ref(''),
            'profile': profile,
            'local_path': path,
            'avatar': to_ref(to_base64(get_image_data('running/default_logo.png')))
        }

    async def get_exp(self):
        records = await self.experiments.all().prefetch_related('user')
        records.extend(await Experiment.filter(user_id__in=self.shared).all().prefetch_related('user'))
        for record in records:
            record.distribution = convert_keys_to_int(record.distribution)
            record.task_names = convert_keys_to_int(record.task_names)
            record.results = convert_keys_to_int(record.results)
        return records

    def get_dict(self):
        profile = {}
        for k, v in profile_dict.items():
            v['default'] = to_ref(copy.deepcopy(self.profile[k]))
            profile[k] = v
        path = {}
        for k, v in path_dict.items():
            v['default'] = to_ref(copy.deepcopy(self.local_path[k]))
            v['name'] = v['name'] + '保存路径'
            path[k] = v

        return {
            'id': to_ref(copy.deepcopy(self.id)),
            'uname': to_ref(copy.deepcopy(self.username)),
            'pwd': to_ref(copy.deepcopy('')),
            'profile': profile,
            'local_path': path,
            'avatar': to_ref(copy.deepcopy(self.avatar))
        }

    @classmethod
    async def login(cls, uname, pwd): # 业务层逻辑
        user = await cls.get_or_none(username=uname)
        if not user:
            return False, '用户名不存在', None
        if not user.verify_password(pwd):
            return False, '密码不正确', None
        else:
            return True, '登录成功', dict(user)

    @classmethod
    async def register(cls, sign_info): # 注册-业务层逻辑
        uname = sign_info['uname'].value
        pwd = sign_info['pwd'].value
        if not uname or uname == '':
            return False, '用户名不能为空', None
        if not pwd or pwd == '':
            return False, '密码不能为空', None
        if await cls.username_exists(uname):
            return False, '用户名已经存在', None
        else:
            profile = {k: v['default'].value for k, v in sign_info['profile'].items()}
            path = {k: v['default'].value for k, v in sign_info['local_path'].items()}
            return await cls.create_new_user(uname, pwd, profile, path, sign_info['avatar'].value)

    async def update(self, k, ref): # 更新-业务层映射
        if k == 'uname':
            return await self.set_username(ref.value)
        elif k == 'pwd':
            return await self.set_password(ref.value)
        elif k == 'profile':
            profile = {k: v['default'].value for k, v in ref.items()}
            return await self.set_profile(profile)
        elif k == 'local_path':
            path = {k: v['default'].value for k, v in ref.items()}
            return await self.set_local_path(path)
        elif k == 'avatar':
            return await self.set_avatar(ref.value)
        else:
            return False, f'user对象没有{k}属性'

    @classmethod
    async def get_unshare(cls, uid):
        # 查询共享用户的ID和用户名
        user = await User.get(id=uid)
        return await User.filter(id__not_in=user.share + [user.id]).values_list('id', 'username')

    @classmethod
    async def set_share(cls, uid: int, share_ids: list):
        print( share_ids)
        async with in_transaction() as transaction:
            try:
                user = await User.get(id=uid)
                original_share_ids = user.share if user.share else []
                # 更新发起共享的用户的`share`属性
                user.share = share_ids
                await user.save()
                # 找出新添加的共享用户ID和被移除的共享用户ID
                added_share_ids = [sid for sid in share_ids if sid not in original_share_ids]
                removed_share_ids = [sid for sid in original_share_ids if sid not in share_ids]
                print(added_share_ids, removed_share_ids)
                # 处理新添加的共享用户
                if len(added_share_ids) > 0:
                    added_users = await User.filter(id__in=added_share_ids).all()
                    for user in added_users:
                        if not user.shared:
                            user.shared = []
                        if uid not in user.shared:
                            user.shared.append(uid)
                            user.shared = json.dumps(user.shared)
                    await User.bulk_update(added_users, fields=['shared'])
                # 处理被移除的共享用户
                if len(removed_share_ids) > 0:
                    print(2)
                    removed_users = await User.filter(id__in=removed_share_ids)
                    for user in removed_users:
                        if user.shared and uid in user.shared:
                            user.shared.remove(uid)
                            user.shared = json.dumps(user.shared)
                    await User.bulk_update(removed_users, fields=['shared'])
            except IntegrityError:
                await transaction.rollback()
                return False, "共享用户设置失败(完整性错误)"
            except Exception as e:
                await transaction.rollback()
                return False, f"共享用户设置失败: {str(e)}"
        return True, "共享用户设置成功"

    @classmethod
    async def get_share(cls, uid):
        # 查询共享用户的ID和用户名
        user = await User.get(id=uid)
        return await User.filter(id__in=user.share).values_list('id', 'username')

    @classmethod
    async def get_shared(cls, uid):
        # 查询共享用户的ID和用户名
        user = await User.get(id=uid)
        return await User.filter(id__in=user.shared).values_list('id', 'username')

    @classmethod
    async def get_all_users(cls):
        """
        获取数据库中所有的User对象。
        """
        return await cls.all()

    @classmethod
    async def delete_user_by_id(cls, user_id: int) -> (bool, str):
        """
        通过用户ID删除用户。
        """
        deleted_count = await cls.filter(id=user_id).delete()
        if deleted_count:
            return True, f"用户{user_id}删除成功"
        else:
            return False, f"用户{user_id}未找到"

    @classmethod
    async def username_exists(cls, username: str) -> bool:
        return await cls.filter(username=username).exists()

    def verify_password(self, plain_password):
        return bcrypt.verify(plain_password, self.password)

    @classmethod
    async def create_new_user(cls, uname: str, pwd: str, profile=None, local_path=None, avatar=None) -> (bool, str):
        """
        尝试创建并插入新的用户记录。
        返回一个元组，包含操作成功与否的布尔值和相应的消息。
        """
        try:
            # 对密码进行加密
            hashed_password = bcrypt.hash(pwd)
            # 创建新用户
            ai_config = {k: v['default'] for k, v in ai_config_dict.items()}
            user = await cls.create(username=uname, password=hashed_password, profile=profile,
                                    local_path=local_path, avatar=avatar, ai_config=ai_config, share=[], shared=[])
            return True, "注册成功", dict(user)
        except IntegrityError:
            # 如果违反了数据库的唯一性约束等
            return False, "注册失败(完整性错误)"
        except Exception as e:
            # 捕获其他可能的异常，并返回错误消息
            return False, f"注册失败(其他错误): {str(e)}"

    @classmethod
    async def get_uname_by_id(cls, uid: int) -> str:
        user = await cls.get_or_none(id=uid)
        return user.username

    async def set_username(self, uname: str) -> (bool, str):
        if not uname or uname == '':
            return False, '用户名不能为空'
        if uname == self.username:
            return False, "用户名不能和上次相同"
        if await User.username_exists(uname):
            return False, "用户名已经存在"

        try:
            self.username = uname
            await self.save()
            # print("Username updated successfully")
            return True, "用户名设置成功"
        except IntegrityError:
            return False, "用户名设置失败(完整性错误)"

    async def set_password(self, pwd: str) -> (bool, str):
        print(pwd)
        if not pwd or pwd == '':
            return False, '密码不能为空'
        if self.verify_password(pwd):
            return False, "密码不能和上次相同"
        try:
            self.password = bcrypt.hash(pwd)
            await self.save()
            return True, "密码修改成功"
        except Exception as e:
            return False, f"密码修改失败: {str(e)}"

    async def set_profile(self, new_profile: Dict) -> (bool, str):
        if self.profile == new_profile:
            return True, "简历无需更新"
        try:
            self.profile = new_profile
            await self.save()
            return True, "简历更新成功"
        except Exception as e:
            return False, f"简历更新失败: {str(e)}"

    async def set_local_path(self, new_local_path: Dict) -> (bool, str):
        if self.local_path == new_local_path:
            return True, "本地路径无需更新"
        try:
            for k, v in new_local_path.items():
                if not os.path.exists(v):
                    return False, f"本地路径设置{k}失败: {v}不存在"
                elif not os.path.isdir(v):
                    return False, f"本地路径{k}设置失败: {v}不是文件夹"
                last_path = self.local_path[k]
                if last_path != v:
                    await move_all_files(last_path, v)

            self.local_path = new_local_path
            await self.save()
            return True, "本地路径更新成功"
        except Exception as e:
            return False, f"本地路径更新失败: {str(e)}"

    @classmethod
    async def set_ai_config(cls, uid, new_ai_config: Dict) -> (bool, str):
        user = await cls.get(id=uid)
        if user.ai_config == new_ai_config:
            return True, "AI配置无需更新"
        try:
            for k, v in new_ai_config.items():
                if k in ['chat_history', 'embedding_files', 'index_files']:
                    last_path = user.ai_config[k]
                    if last_path != v:
                        if not os.path.exists(v):
                            return False, f"AI配置{k}设置失败: {v}不存在"
                        elif not os.path.isdir(v):
                            return False, f"AI配置{k}设置失败: {v}不是目录"
                        await move_all_files(last_path, v)

            user.ai_config = new_ai_config
            await user.save()
            return True, "AI配置更新更新成功"
        except Exception as e:
            return False, f"AI配置更新失败: {str(e)}"

    async def set_avatar(self, new_avatar: str) -> (bool, str):
        if self.avatar == new_avatar:
            return True, "头像无需更新"
        try:
            self.avatar = new_avatar
            await self.save()
            return True, "头像更新成功"
        except Exception as e:
            return False, f"头像更新失败: {str(e)}"
