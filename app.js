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
const zerorpc = require("zerorpc");
const PythonConnector = require('./pythonConnector.js');

const fileUpload = require('express-fileupload');
const path = require('path');
var upload = multer()
app.use(bodyParser.json());
app.use(cors())
app.use(bodyParser.urlencoded());
app.use(bodyParser.urlencoded({ extended: true }));


// RPC Setup
const constLargeEnoughHeartbeat = 60 * 60 * 24 * 30 * 12 * 1000 // 1 Year
clientOptions = {
    "timeout": 120,
    "heartbeatInterval": constLargeEnoughHeartbeat,
}
var client = new zerorpc.Client(clientOptions);
client.connect("tcp://127.0.0.1:4242");

// Express Setup
app.listen(25565, () => console.log("server listening on port 5000"));



app.get('/test', (req, res) => {
    client.invoke("test", function(err, _res, more) {
        console.log(_res);
    });
})

app.post('/api/v1/finetune', upload.any(), (req, res) => {
    req.setTimeout(500000);
    // console.log(req.files);
    let id = genRanHex(6)
    console.log(req.body);
    // console.log(req.files);
    req.files.map((file, i) => {
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
        }
        let fileType = getFileType(String(mimetype));
        fse.outputFile(`saves/${id}/train_noise/${fieldname}/${originalname}`, buffer)
            .then(() => {
                console.log("saved", `saves/${id}/train_noise/${fieldname}/${originalname}`)
                if (i == req.files.length - 1) {
                    console.log("Got called here");
                    let type = req.body.type
                    let accuracy = req.body.type == "pneumonia_id" ? 0.8 : 0.75
                    let nlt = req.body.nlt
                    let nle = req.body.nle
                    client.invoke("fine_tune", type, `/home/jason/Hackalytics-backend/saves/${id}`, accuracy, nlt, nle, function(err, _res, more) {
                        if(err) {
                            console.log(err);
                            console.log("something happened man");
                            return;
                        };
                        console.log(_res)
                        res.send(_res)
                    })         
                }
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
