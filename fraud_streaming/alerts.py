# alerts.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email_config import SENDER_EMAIL, APP_PASSWORD, RECEIVER_EMAIL

def send_fraud_alert(row):
    """
    Sends fraud alert email when a transaction is flagged as fraudulent.
    """

    subject = f"⚠️ Fraud Alert — Transaction ID: {row.get('Transaction_ID', 'Unknown')}"
    
    body = f"""
    <h2>🚨 Fraudulent Transaction Detected</h2>

    <p><b>Transaction ID:</b> {row.get('Transaction_ID')}</p>
    <p><b>User ID:</b> {row.get('User_ID')}</p>
    <p><b>Amount:</b> {row.get('Transaction_Amount')}</p>
    <p><b>Location:</b> {row.get('Location')}</p>
    <p><b>Merchant Category:</b> {row.get('Merchant_Category')}</p>

    <p>This transaction has been flagged as <b>HIGH RISK</b> by the Fraud Detection System.</p>

    <p>Please review immediately.</p>

    <hr>
    <p>Automated Fraud Monitoring System</p>
    """

    # Construct email
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()

        print(f"[ALERT EMAIL SENT] Fraud detected — Transaction ID: {row.get('Transaction_ID')}")

    except Exception as e:
        print(f"[ERROR] Could not send email alert: {e}")
