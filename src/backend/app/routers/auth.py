from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .. import models, schemas, auth, dependencies

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)


@router.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(dependencies.get_db)):
    """
    Yeni kullanıcı kaydı:
    - hospital_code OPSİYONEL.
    - Eğer varsa, hastaneye bağlar (STAFF için).
    - Yoksa, bağımsız kullanıcı oluşturur (ADMIN için ideal).
    """
    hospital_id = None
    
    # 1) Hastane Kodu varsa kontrol et
    if user.hospital_code:
        db_hospital = (
            db.query(models.Hospital)
            .filter(models.Hospital.unique_code == user.hospital_code)
            .first()
        )
        if not db_hospital:
            raise HTTPException(status_code=400, detail="Invalid hospital code")
        hospital_id = db_hospital.id

    # 2) Email kontrolü
    existing_user = (
        db.query(models.User)
        .filter(models.User.email == user.email)
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # 3) Kullanıcı oluştur
    hashed_password = auth.get_password_hash(user.password)

    db_user = models.User(
        email=user.email,
        password_hash=hashed_password,
        role=user.role, # Use role from request (defaults to TECH_STAFF)
        hospital_id=hospital_id, # Can be None
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/login", response_model=schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(dependencies.get_db),
):
    """
    Login endpoint:
    - form_data.username = email
    - form_data.password = şifre
    - doğruysa JWT üretir, yanlışsa 401 döner
    """
    user = (
        db.query(models.User)
        .filter(models.User.email == form_data.username)
        .first()
    )

    if user is None or not auth.verify_password(
        form_data.password, user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(
        minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    access_token = auth.create_access_token(
        data={
            "sub": str(user.id),
            "hospital_id": user.hospital_id,
            # models.User'da role yoksa 500 patlamasın diye getattr kullandım
            "role": getattr(user, "role", "admin"),
        },
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=schemas.UserResponse)
def read_users_me(
    current_user: models.User = Depends(dependencies.get_current_user),
):
    """
    Access token ile giriş yapmış kullanıcının bilgilerini döner.
    """
    return current_user
