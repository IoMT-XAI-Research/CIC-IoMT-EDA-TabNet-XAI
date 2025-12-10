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
    - hospital_code ile hastaneyi bulur (yoksa otomatik oluşturur)
    - email daha önce alınmışsa 400 döner
    - şifreyi hashleyip User kaydı oluşturur
    """
    # 1) Hastaneyi bul veya yoksa oluştur
    db_hospital = (
        db.query(models.Hospital)
        .filter(models.Hospital.code == user.hospital_code)
        .first()
    )

    if db_hospital is None:
        # Demo için: hospital_code yoksa otomatik hastane oluştur
        db_hospital = models.Hospital(
            name="Default Hospital",
            code=user.hospital_code,
        )
        db.add(db_hospital)
        db.commit()
        db.refresh(db_hospital)

    # 2) Email daha önce alınmış mı?
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

    # 3) Şifreyi hashle ve kullanıcıyı kaydet
    hashed_password = auth.get_password_hash(user.password)

    db_user = models.User(
        email=user.email,
        password_hash=hashed_password,
        hospital_id=db_hospital.id,
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
