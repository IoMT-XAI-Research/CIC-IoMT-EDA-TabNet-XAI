import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ..core.config import settings

def send_reset_email(to_email: str, token: str):
    subject = "IoMT Şifre Sıfırlama"
    # Link Updated to use dynamic SSR route
    reset_link = f"{settings.SERVER_HOST}/auth/reset-password?token={token}"
    
    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="background-color: #f7f7f7; padding: 20px; border-radius: 5px;">
                <h2 style="color: #00E676;">IoMT Şifre Sıfırlama Talebi</h2>
                <p>Merhaba,</p>
                <p>Hesabınız için şifre sıfırlama talebi aldık. Aşağıdaki butona tıklayarak yeni şifrenizi belirleyebilirsiniz:</p>
                <br>
                <a href="{reset_link}" style="background-color: #00E676; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Şifremi Sıfırla</a>
                <br><br>
                <p>VEYA aşağıdaki linke tıklayın:</p>
                <p><a href="{reset_link}">{reset_link}</a></p>
                <br>
                <p style="font-size: 12px; color: #999;">Eğer bu talebi siz yapmadıysanız, lütfen bu emaili dikkate almayınız.</p>
            </div>
        </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = settings.EMAILS_FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(settings.EMAILS_FROM_EMAIL, to_email, text)
        server.quit()
        print(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
