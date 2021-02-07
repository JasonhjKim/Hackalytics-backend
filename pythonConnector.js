class PythonConnector {
    static server() {
        if (!PythonConnector.connected) {
            console.log('PythonConnector – making a new connection to the python layer');
            PythonConnector.zerorpcProcess = spawn('python3', ['-u', path.join(__dirname, 'PythonServer.py')]);
            PythonConnector.zerorpcProcess.stdout.on('data', function(data) {
                console.info('python:', data.toString());
            });
            PythonConnector.zerorpcProcess.stderr.on('data', function(data) {
                console.error('python:', data.toString());
            });
            PythonConnector.zerorpc = new zerorpc.Client({'timeout': TIMEOUT, 'heartbeatInterval': TIMEOUT*1000});
            PythonConnector.zerorpc.connect('tcp://' + IP + ':' + PORT);
            PythonConnector.connected = true;
        }
        return PythonConnector.zerorpc;
    }
}