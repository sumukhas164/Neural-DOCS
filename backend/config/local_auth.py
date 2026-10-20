"""A single non-admin identity for loopback-only development requests."""

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_ADDRESSES = {"127.0.0.1", "::1"}
DEMO_USERNAME = "__neural_docs_local_demo__"


def is_local_demo_request(request):
    if not (settings.DEBUG and settings.LOCAL_DEMO_MODE
            and request.META.get("REMOTE_ADDR") in LOCAL_ADDRESSES):
        return False
    host = request.get_host()
    host = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.split(":", 1)[0]
    return host in LOCAL_HOSTS


class LocalDemoAuthentication(BaseAuthentication):
    def authenticate(self, request):
        if not is_local_demo_request(request):
            return None
        user, _ = get_user_model().objects.get_or_create(
            username=DEMO_USERNAME,
            defaults={"password": make_password(None), "is_staff": False, "is_superuser": False},
        )
        if user.has_usable_password() or user.is_staff or user.is_superuser:
            raise AuthenticationFailed("The reserved local demo account conflicts with an existing user")
        return user, None
