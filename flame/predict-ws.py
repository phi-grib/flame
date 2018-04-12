#! -*- coding: utf-8 -*-

##    Description    Flame web-service
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import cherrypy
import json
import shutil
import tempfile

from predict import Predict
import manage
import util.utils as utils

PARTNER_ID = 'UPF'
PARTNER_WEB ="http://phi.upf.edu"
ADMIN_NAME = 'Manuel Pastor'
ADMIN_EMAIL = 'manuel.pastor@upf.edu'


class FlamePredict(object):
    @cherrypy.expose
    def index(self):
        return open('./templates/index.html')

    @cherrypy.expose
    def upload(self):

        filename    = os.path.basename(cherrypy.request.headers['x-filename'])
        temp_dir    = os.path.basename(cherrypy.request.headers['temp-dir'])

        path = tempfile.gettempdir()+'/'+temp_dir
        os.mkdir (path)
 
        destination = os.path.join(path, filename)
        with open(destination, 'wb') as f:
            shutil.copyfileobj(cherrypy.request.body, f)

@cherrypy.expose
class FlamePredictWS(object):

    @cherrypy.tools.accept(media='text/plain')

    def POST(self, ifile, model, version, temp_dir):

        ifile = tempfile.gettempdir()+'/'+temp_dir+'/'+ifile
        
        if version[:3]=='ver': 
            version = int(version[-6:]) ## get the numbers

        version = utils.intver(version)

        # try:
        #     predict = Predict(ifile, model, version)
        #     success, results = predict.run()
        # except:
        #     raise cherrypy.HTTPError(500)
            
        predict = Predict(ifile, model, version)
        success, results = predict.run()

        return results

@cherrypy.expose
class FlameInfoWS(object):

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        data = { "provider": PARTNER_ID,
                 "homepage": PARTNER_WEB,
                 "admin": ADMIN_NAME,
                 "admin_email": ADMIN_EMAIL
                 }
        return json.dumps(data)

@cherrypy.expose
class FlameDirWS(object):

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):

        success, results = manage.action_dir()

        if not success:
            return "no model found"
        return results

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': False,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/info': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        },
        '/dir': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        },
        '/predict': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './public',
        },
        'global' : {
            'server.socket_host' : '0.0.0.0',
            'server.socket_port' : 8080,
            'server.thread_pool' : 8,
        }
    }

    webapp = FlamePredict()
    webapp.info = FlameInfoWS()
    webapp.dir = FlameDirWS()
    webapp.predict = FlamePredictWS()

    cherrypy.quickstart(webapp, '/', conf)
