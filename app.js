const express = require('express');
const app = express();
// const zerorpc = require("zerorpc");
// const PythonConnector = require('./PythonConnector.js');
const exec = require('child_process').exec;


const bodyParser = require('body-parser');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const fse = require('fs-extra');

const fileUpload = require('express-fileupload');
const path = require('path');
var upload = multer()
app.use(bodyParser.json());
app.use(cors())
app.options('*', cors());
app.use(bodyParser.urlencoded());
app.use(bodyParser.urlencoded({ extended: true }));
// app.use(function(req, res, next) {
//    res.header("Access-Control-Allow-Origin", "*");
//    res.header('Access-Control-Allow-Methods', 'DELETE, PUT, GET, POST');
//    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
//    next();
// });
// app.setTimeout(120000)

// const socketIO = require("socket.io");
// const server = require("http").createServer(app);
// const io = socketIO(server, {origins: 'http://localhost:3000', methods: ["GET", "POST"]}).listen(server);


// app.use(upload.array()); 


// var commands = [
//     // 'source /home/jason/anaconda3/etc/profile.d/conda.sh',
//     'conda info',
//     'python /home/jason/playground/main.py',

// ]

// exec(commands.join(' & '),
//     function(error, stdout, stderr){
//         console.log(error)
//         console.log(stdout)
//         console.log(stderr)
//     }
// );


// RPC Setup
// var client = new zerorpc.Client();
// client.connect("tcp://127.0.0.1:4242");

// Express Setup
app.listen(5000, () => console.log("server listening on port 5000"));



app.get('/test', (req, res) => {
    client.invoke("test", function(err, _res, more) {
        console.log(_res);
    });
})

app.post('/api/v1/finetune', upload.any(), (req, res) => {
    // console.log(req.files);
    let id = genRanHex(6)
    req.files.map((file) => {
        const {fieldname, originalname, mimetype, buffer} = file;
        if(fieldname == "py") {
            console.log("found py");
            fse.outputFile(`saves/${id}/extension.py`, buffer)
                .then(() => {
                    console.log(`saves/${id}/extension.py`)
                })
                .catch((err) => {
                    console.log(err)
                })
            return;
        }
        console.log(fieldname, originalname, mimetype)
        let fileType = getFileType(String(mimetype));
        fse.outputFile(`saves/${id}/${fieldname}/${originalname}`, buffer)
            .then(() => {
                console.log("saved", `saves/${id}/${fieldname}/${originalname.py}`)
                setTimeout(() => {
                    res.send("done")
                }, 120000);
                // client.invoke("finetune", "model path", "image path", function(err, res, more) {
                    //
                // }) 
            })
            .catch(err => {
                console.log(err)
            })
        })

    
})

const genRanHex = size => [...Array(size)].map(() => Math.floor(Math.random() * 16).toString(16)).join('');
const getFileType = (ft) => {
    return ft.substring(ft.indexOf("/"), ft.length-1);
}




// client.invoke("hello", "RPC", function(error, res, more) {
//     console.log(res);
// });

// client.invoke("model", function(error, res, more){
//     console.log(res);
// });
