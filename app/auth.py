from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from starlette import status
from typing import Annotated

from app.config import settings
from app.models import APIUser
from app.pydantic_models import CreateUserRequest, Token


auth_app = APIRouter(prefix='/auth', tags=['Auth'])
bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oath2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')



@auth_app.post('/', status_code=status.HTTP_201_CREATED)
async def create_user(create_user_req: CreateUserRequest):
    user = APIUser(username=create_user_req.username, 
                   hashed_password=bcrypt_context.hash(create_user_req.password),
                   role=create_user_req.role)
    print(f'user={user}')
    await user.save()
    
    return {'user': user} 


@auth_app.post('/token', response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Could not validate user')
    token = create_access_token(user.username, user.id, timedelta(minutes=2))

    return {'access_token': token, 'token_type': 'bearer'} 


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
    encode = {'sub': username, 'id': user_id}
    expires = datetime.now(timezone.utc) + expires_delta
    encode.update({'exp': expires})

    return jwt.encode(encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(token: Annotated[str, Depends(oath2_bearer)]):
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        if username is None or user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail='Could not validate user')
        return {'username': username, 'id': user_id}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Could not validate user')
    

user_dependency = Annotated[dict, Depends(get_current_user)]
