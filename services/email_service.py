import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from models import EmailRequest

def send_job_summary_email(request: EmailRequest):
    """
    Sends a formatted email with the job summary to the specified address.
    """
    sender_email = os.getenv("EMAIL_HOST_USER")
    sender_password = os.getenv("EMAIL_HOST_PASSWORD")
    
    if not sender_email or not sender_password:
        raise ValueError("Email credentials not configured in environment variables.")

    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = "Your JobScout AI - Job Match Summary"
    message["From"] = sender_email
    message["To"] = request.email

    # Build HTML Content
    jobs_html = ""
    for job in request.matched_jobs:
        jobs_html += f"""
        <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;">
            <h3 style="margin-top: 0; color: #3b82f6;">{job.title}</h3>
            <p><strong>Company:</strong> {job.company}</p>
            <p><strong>Location:</strong> {job.location or 'Remote'}</p>
            <p><strong>Match Score:</strong> <span style="color: #059669; font-weight: bold;">{job.score}/100</span></p>
            <p><strong>Reasoning:</strong> {job.reasoning}</p>
            <a href="{job.url}" style="display: inline-block; padding: 10px 20px; background-color: #3b82f6; color: white; text-decoration: none; border-radius: 5px;">View Position</a>
        </div>
        """

    html = f"""
    <html>
        <body style="font-family: sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #3b82f6; text-align: center;">JobScout AI Results</h1>
                <p>Hello,</p>
                <p>Based on your CV, our autonomous agent has found the following job opportunities for you:</p>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                {jobs_html}
                <p style="font-size: 0.8em; color: #777; margin-top: 30px; text-align: center;">
                    © 2026 JobScout AI • Built for SMIT Hackathon
                </p>
            </div>
        </body>
    </html>
    """

    part = MIMEText(html, "html")
    message.attach(part)

    # Send email
    try:
        # Using port 587 with STARTTLS for better compatibility
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, request.email, message.as_string())
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise e
