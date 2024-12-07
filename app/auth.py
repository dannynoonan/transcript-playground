from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from starlette import status
from typing import Annotated

from app.config import settings
from app.models import APIUser


bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oath2_scheme = OAuth2PasswordBearer(tokenUrl='auth/token')


async def authenticate_user(username: str, password: str) -> APIUser|None:
    user = await APIUser.filter(username=username).first()
    if not user:
        return None
    # user_pyd = await pymod.APIUserPydantic.from_tortoise_orm(user)
    # print(f'type(user_pyd)={type(user_pyd)} user_pyd={user_pyd}')
    if not bcrypt_context.verify(password, user.hashed_password):
        return None
    return user


def create_access_token(username: str, user_id: int, expires_delta: timedelta) -> str:
    to_encode = {'sub': username, 'id': user_id}
    expires = datetime.now(timezone.utc) + expires_delta
    # to_encode['exp'] = expires
    to_encode.update({'exp': expires})

    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(token: Annotated[str, Depends(oath2_scheme)]):
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        if username is None or user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail='Could not validate user')
        # user = await APIUser.filter(username=username).first()
        return {'username': username, 'id': user_id}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Could not validate user')
    

user_dependency = Annotated[dict, Depends(get_current_user)]


def exit_if_unauthorized(user: user_dependency, level: str = None):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    if level and level == 'admin':
        if user['username'] != settings.api_admin_user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"User `{user['username']}` is not authorized to use this endpoint")


ADMIN_USER = authenticate_user(settings.api_admin_user, settings.api_admin_password)
