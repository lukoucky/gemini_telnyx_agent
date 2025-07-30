import telnyx
import os
import dotenv
import validators

dotenv.load_dotenv()


TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
NGROK_URL = os.getenv("NGROK_URL")
WEBHOOK_URL = f"{NGROK_URL}/webhook"
PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER")
APPLICATION_NAME = os.getenv("APPLICATION_NAME")
OUTBOUND_PROFILE_NAME = f"{APPLICATION_NAME} Voice Profile"


def update_application_webhook(app_id):
    # Validate the new webhook URL
    if not validators.url(WEBHOOK_URL) or not WEBHOOK_URL.startswith("https://"):
        raise Exception(f"Error: Webhook URL must be a valid HTTPS URL. Provided URL: {WEBHOOK_URL}")
    
    # Get current state first
    current_app = telnyx.CallControlApplication.retrieve(app_id)
    old_webhook_url = current_app.webhook_event_url
    if old_webhook_url == WEBHOOK_URL:
        print("Old webhook URL is same as new one - no need to update")
        return
    
    update_params = {
        "application_name": current_app.application_name,
        "webhook_event_url": WEBHOOK_URL,
        "webrtc_enabled": True
    }
    telnyx.CallControlApplication.modify(app_id, **update_params)
    print(f"Updated Webhook URL from {old_webhook_url} to {WEBHOOK_URL}")
    

def create_voice_application():
    # 1) Check if app already exist
    apps = telnyx.CallControlApplication.list()
    for app in apps:
        if APPLICATION_NAME == app.application_name:
            print(f"Found application with name {APPLICATION_NAME} (ID: {app.id}), updating already existing app")
            update_application_webhook(app.id)
            return str(app.id)

    # 2) If app does not exist - create it
    application = telnyx.CallControlApplication.create(
        application_name=APPLICATION_NAME,
        webhook_event_url=WEBHOOK_URL,
        webhook_event_failover_url="",
        webrtc_enabled=True
    )
    print(f"Created application: {application.application_name} (ID: {application.id})")
    return str(application.id)


def get_phone_number_id():
    numbers = telnyx.PhoneNumber.list()
    phone_number_id = None
    for number in numbers:
        if number.phone_number == PHONE_NUMBER:
            print(f"Found phone number {number.phone_number} with ID: {number.id}")
            phone_number_id = str(number.id)

    if phone_number_id is None:
        print(f"Phone number {PHONE_NUMBER} not found in telnyx")
        print(f"Make sure you set TELNYX_PHONE_NUMBER env variable in correct format +12345678910")
        raise Exception(f"Wrong phone number {PHONE_NUMBER}")
    return str(phone_number_id)


def update_phone_number_application(app_id):
    phone_number_id = get_phone_number_id()
    number = telnyx.PhoneNumber.retrieve(phone_number_id)
    old_app_id = number.connection_id
    if old_app_id == app_id:
        print("Old app id in phone number is the same as new app id - no need to update")
        return
    
    update_params = {
        "connection_id": app_id
    }
    
    telnyx.PhoneNumber.modify(phone_number_id, **update_params)
    print(f"Updated app id in phone number from {old_app_id} to {app_id}")


def create_outbound_voice_profile(app_id):
    profiles = telnyx.OutboundVoiceProfile.list()
    for profile in profiles:
        if profile.name == OUTBOUND_PROFILE_NAME:
            print(f"Outbound voice profile {OUTBOUND_PROFILE_NAME} already exist, removing old profile")
            profile.delete()
            break

    # Create new outbound voice profile and link it to the application
    profile = telnyx.OutboundVoiceProfile.create(
        name=OUTBOUND_PROFILE_NAME,
        call_control_application_id=app_id
    )
    print(f"Created outbound voice profile: {profile.name} (ID: {profile.id})")


if __name__ == "__main__":
    if TELNYX_API_KEY is None or NGROK_URL is None or PHONE_NUMBER is None or APPLICATION_NAME is None:
        raise Exception("Missing one of following env variables: TELNYX_API_KEY, NGROK_URL, PHONE_NUMBER, APPLICATION_NAME")
    telnyx.api_key = TELNYX_API_KEY
    app_id = create_voice_application()
    update_phone_number_application(app_id)
    create_outbound_voice_profile(app_id)
