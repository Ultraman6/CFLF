import copy
import os
from sqlite3 import IntegrityError
from ex4nicegui import to_ref
from tortoise import fields, models
from passlib.hash import bcrypt
from typing import Dict
from visual.parts.constant import profile_dict, path_dict, ai_config_dict


class User(models.Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=255, unique=True)  # 用户名作为登录凭据
    password = fields.CharField(max_length=255)
    profile = fields.JSONField(null=True)  # 包含学历和研究方向的JSON字段
    local_path = fields.JSONField(null=True)  # 包含三种本地路径的JSON字段
    avatar = fields.TextField(null=True)   # 用户头像路径
    ai_config = fields.JSONField(null=True)

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
            'avatar': to_ref('https://nicegui.io/logo_square.png')
        }

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
    async def login(cls, uname, pwd):
        user = await cls.get_or_none(username=uname)
        if not user:
            return False, '用户名不存在', None
        if not user.verify_password(pwd):
            return False, '密码不正确', None
        else:
            return True, '登录成功', dict(user)

    @classmethod
    async def register(cls, sign_info):
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

    async def update(self, k, ref):
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
            user = await cls.create(username=uname, password=hashed_password, profile=profile, local_path=local_path, avatar=avatar, ai_config=ai_config)
            return True, "注册成功", dict(user)
        except IntegrityError:
            # 如果违反了数据库的唯一性约束等
            return False, "注册失败(完整性错误)"
        except Exception as e:
            # 捕获其他可能的异常，并返回错误消息
            return False, f"注册失败(其他错误): {str(e)}"

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
            self.local_path = new_local_path
            await self.save()
            return True, "本地路径更新成功"
        except Exception as e:
            return False, f"本地路径更新失败: {str(e)}"

    async def set_avatar(self, new_avatar: str) -> (bool, str):
        if self.avatar == new_avatar:
            return True, "头像无需更新"
        try:
            self.avatar = new_avatar
            await self.save()
            return True, "头像更新成功"
        except Exception as e:
            return False, f"头像更新失败: {str(e)}"
