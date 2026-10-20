from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .local_auth import is_local_demo_request


class RuntimeConfigView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"local_demo_mode": is_local_demo_request(request)})
