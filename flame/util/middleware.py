import re
import os
import logging
import json
from django.conf import settings
from django.http.response import JsonResponse
from django.utils.deprecation import MiddlewareMixin
from keycloak import KeycloakOpenID
from rest_framework.exceptions import PermissionDenied, AuthenticationFailed, NotAuthenticated
from keycloak.exceptions import KeycloakError

logger = logging.getLogger(__name__)

config_in_settings = settings.APPLICATION_CONFIG

def get_application_config_for_key(var_key):
    
    # 1. environment variables
    config_val = os.environ.get(var_key)
    if config_val is not None:
        # print('os:', var_key, config_val)
        return config_val
    
    # 2. config file in /secrets

    #   "realm": "KH",
    #   "auth-server-url": "http://localhost:8090/auth",
    #   "ssl-required": "external",
    #   "resource": "knowledge-hub",
    #   "credentials": {
    #     "secret": "**********"
    #   }

    # if os.path.isfile('/secrets/keycloak.json'):
    #     with open('/secrets/keycloak.json') as f:
    #         keycloak_file = json.load(f)
        
    #     translate= {'KEYCLOAK_SERVER_URL':'auth-server-url',
    #                 'KEYCLOAK_REALM':'realm',
    #                 'KEYCLOAK_CLIENT_ID':'resource',
    #                 'KEYCLOAK_CLIENT_SECRET_KEY':'secret'}

    #     if (var_key) in translate:
    #         var_key_trans = translate[var_key]
    #         if (var_key_trans) in keycloak_file:
    #             return keycloak_file[var_key]
    #         elif (var_key_trans == 'secret'):
    #             if 'credentials' in keycloak_file:
    #                 return keycloak_file['credentials']['secret']

    # 3. hardcoded in APLICATION_CONFIG (django)

    if var_key in config_in_settings:
        config_val = config_in_settings[var_key]
        # print('settings:', var_key, config_val)
        return config_val
    else:
        return None

class KeycloakMiddleware(MiddlewareMixin):

    def __init__(self, get_response):
        """

        :param get_response:
        """

        self.config = settings.APPLICATION_CONFIG

        # Read configurations
        try:
            self.server_url = get_application_config_for_key('KEYCLOAK_SERVER_URL')
            self.client_id = get_application_config_for_key('KEYCLOAK_CLIENT_ID')
            self.realm = get_application_config_for_key('KEYCLOAK_REALM')
        except KeyError as e:
            raise Exception("KEYCLOAK_SERVER_URL, KEYCLOAK_CLIENT_ID or KEYCLOAK_REALM not found.")

        self.client_secret_key = get_application_config_for_key('KEYCLOAK_CLIENT_SECRET_KEY')
        self.default_access = get_application_config_for_key('KEYCLOAK_DEFAULT_ACCESS')
        self.method_validate_token = get_application_config_for_key('KEYCLOAK_METHOD_VALIDATE_TOKEN')
        self.keycloak_authorization_config = get_application_config_for_key('KEYCLOAK_AUTHORIZATION_CONFIG')

        # Create Keycloak instance
        self.keycloak = KeycloakOpenID(server_url=self.server_url,
                                       client_id=self.client_id,
                                       realm_name=self.realm,
                                       client_secret_key=self.client_secret_key,
                                       )

        # Read policies
        if self.keycloak_authorization_config:
            self.keycloak.load_authorization_config(self.keycloak_authorization_config)

        # Django
        self.get_response = get_response

        print("KEYCLOAK_SERVER URL", get_application_config_for_key('KEYCLOAK_SERVER_URL'))
        print("KEYCLOAK_CLIENT_ID URL", get_application_config_for_key('KEYCLOAK_CLIENT_ID'))
        print("KEYCLOAK_REALM URL", get_application_config_for_key('KEYCLOAK_REALM'))


    @property
    def keycloak(self):
        return self._keycloak

    @keycloak.setter
    def keycloak(self, value):
        self._keycloak = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def server_url(self):
        return self._server_url

    @server_url.setter
    def server_url(self, value):
        self._server_url = value

    @property
    def client_id(self):
        return self._client_id

    @client_id.setter
    def client_id(self, value):
        self._client_id = value

    @property
    def client_secret_key(self):
        return self._client_secret_key

    @client_secret_key.setter
    def client_secret_key(self, value):
        self._client_secret_key = value

    @property
    def client_public_key(self):
        return self._client_public_key

    @client_public_key.setter
    def client_public_key(self, value):
        self._client_public_key = value

    @property
    def realm(self):
        return self._realm

    @realm.setter
    def realm(self, value):
        self._realm = value

    @property
    def keycloak_authorization_config(self):
        return self._keycloak_authorization_config

    @keycloak_authorization_config.setter
    def keycloak_authorization_config(self, value):
        self._keycloak_authorization_config = value

    @property
    def method_validate_token(self):
        return self._method_validate_token

    @method_validate_token.setter
    def method_validate_token(self, value):
        self._method_validate_token = value

    def __call__(self, request):
        """
        :param request:
        :return:
        """
        return self.get_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        """
        Validate only the token introspect.
        :param request: django request
        :param view_func:
        :param view_args: view args
        :param view_kwargs: view kwargs
        :return:
        """

        # print ('DEBUG:', request)

        # do not block the access to root!
        if request.path_info == '/':
            logger.debug('** exclude path found, skipping')
            return None
        
        # CONTINGENCY SOLUTION!!! review API to avoid this
        # if request.path_info[-7:] == '/series':
        #     return None

        whitelist = [
             '/api/v1/api',
             '/api/v1/alive',
             '/api/v1/ready',
            #  '/oformat/WORD/documentation',
            #  '/oformat/EXCEL/documentation', 
            #  '/export_download'
        ]
        for iwhite in whitelist:
            if request.path_info.endswith(iwhite):
                print ('skipped:', request.path_info)
                return None

        if hasattr(settings, 'KEYCLOAK_BEARER_AUTHENTICATION_EXEMPT_PATHS'):
            path = request.path_info.lstrip('/')

            if any(re.match(m, path) for m in
                   settings.KEYCLOAK_BEARER_AUTHENTICATION_EXEMPT_PATHS):
                logger.debug('** exclude path found, skipping')
                return None

        try:
            roles = view_func.cls.roles
        except AttributeError as e:
            return JsonResponse({"detail": NotAuthenticated.default_detail}, status=NotAuthenticated.status_code)

        if 'HTTP_AUTHORIZATION' not in request.META:
            return JsonResponse({"detail": NotAuthenticated.default_detail}, status=NotAuthenticated.status_code)

        auth_header = request.META.get('HTTP_AUTHORIZATION').split()
        token = auth_header[1] if len(auth_header) == 2 else auth_header[0]

        try:
            userinfo = self.keycloak.userinfo(token)
        except KeycloakError as e:
            return JsonResponse({"detail": AuthenticationFailed.default_detail}, status=AuthenticationFailed.status_code)

        has_role = True
        for role in roles:
            if role not in userinfo['groups']:
                has_role = False

        if has_role:
            return None

        # User Permission Denied
        return JsonResponse({"detail": PermissionDenied.default_detail}, status=PermissionDenied.status_code)