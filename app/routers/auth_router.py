from datetime import timedelta
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from starlette import status
from typing import Annotated

from app.auth import bcrypt_context, authenticate_user, create_access_token, exit_if_unauthorized
from app.models import APIUser
from app.pydantic_models import CreateUserRequest, Token


auth_app = APIRouter(tags=['Auth'])


@auth_app.get("/")
def root():
    return {"message": "Welcome to transcript playground! To use the 'read' API please visit '/docs', invoke the '/auth/create_user' endpoint to set up credentials, then click 'Authorize'."}


@auth_app.post('/auth/create_user', status_code=status.HTTP_201_CREATED)
async def create_user(create_user_req: CreateUserRequest):
    api_user = APIUser(username=create_user_req.username, 
                       hashed_password=bcrypt_context.hash(create_user_req.password),
                       role='read')
    # print(f'api_user={api_user}')
    await api_user.save()
    
    return {'api_user': api_user} 


@auth_app.post('/auth/token', response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    api_user = await authenticate_user(form_data.username, form_data.password)
    exit_if_unauthorized(api_user)
    token = create_access_token(api_user.username, api_user.id, timedelta(minutes=20))

    return {'access_token': token, 'token_type': 'bearer'} 
