from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
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
            "role": user.role.value,
        },
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer", "role": user.role.value}


@router.get("/me", response_model=schemas.UserResponse)
def read_users_me(
    current_user: models.User = Depends(dependencies.get_current_user),
):
    """
    Access token ile giriş yapmış kullanıcının bilgilerini döner.
    """
    return current_user

# --- AUTH & PASSWORD RESET ---

@router.post("/forgot-password")
def forgot_password(
    request: schemas.ForgotPasswordRequest,
    db: Session = Depends(dependencies.get_db)
):
    """
    Forgot Password Endpoint (Prod Ready)
    1. Check email
    2. Generate Token
    3. Send REAL SMTP Email
    """
    user = db.query(models.User).filter(models.User.email == request.email).first()
    if not user:
        # User not found: Return 200 to prevent user enumeration
        return {"msg": "If email exists, reset link sent."}

    # Generate Reset Token (Short expiry: 15 mins)
    reset_token = auth.create_access_token(
        data={"sub": str(user.id), "type": "reset"},
        expires_delta=timedelta(minutes=15)
    )

    # Convert Pydantic Settings to use its SMTP config, OR import the new utility
    # We will use the new utils/email.py
    from ..utils import email as email_utils
    email_sent = email_utils.send_reset_email(request.email, reset_token)

    if not email_sent:
        raise HTTPException(status_code=500, detail="Email sending failed.")

    return {"msg": "Password reset email sent."}


# --- SSR PASSWORD RESET ---
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form
import os

# Dynamic template directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

@router.get("/reset-password")
def get_reset_password_page(request: Request, token: str):
    """
    SSR: Render the Reset Password HTML Page
    """
    return templates.TemplateResponse("reset_password.html", {"request": request, "token": token})

@router.post("/reset-password", response_class=HTMLResponse)
async def reset_password_confirm(
    request: Request,
    token: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(dependencies.get_db)
):
    """
    SSR Logic: Handle Form Submission
    """
    error = None
    success = None
    
    if new_password != confirm_password:
        error = "Şifreler eşleşmiyor."
        return templates.TemplateResponse("reset_password.html", {"request": request, "token": token, "error": error})

    try:
        # Verify Token
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")
        
        if not user_id or token_type != "reset":
             error = "Geçersiz veya süresi dolmuş token."
             return templates.TemplateResponse("reset_password.html", {"request": request, "token": token, "error": error})

        # Get User
        user = db.query(models.User).filter(models.User.id == int(user_id)).first()
        if not user:
             error = "Kullanıcı bulunamadı."
             return templates.TemplateResponse("reset_password.html", {"request": request, "token": token, "error": error})

        # Update Password
        user.password_hash = auth.get_password_hash(new_password)
        db.commit()
        success = "Şifreniz başarıyla güncellendi! Giriş yapabilirsiniz."
        
    except auth.JWTError:
        error = "Geçersiz veya süresi dolmuş token."
    except Exception as e:
        error = f"Bir hata oluştu: {str(e)}"

    return templates.TemplateResponse("reset_password.html", {
        "request": request, 
        "token": token, 
        "error": error, 
        "success": success
    })

# Keep API version for potential mobile usage (optional)
@router.post("/reset-password")
def reset_password(
    request: schemas.PasswordResetRequest,
    db: Session = Depends(dependencies.get_db)
):
    # ... (existing API logic if needed, but we focus on SSR now)
    pass
