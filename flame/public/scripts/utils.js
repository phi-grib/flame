//     Description    Flame JavaScript utils
//
//     Authors:       Manuel Pastor (manuel.pastor@upf.edu)
// 
//     Copyright 2018 Manuel Pastor
// 
//     This file is part of Flame
// 
//     Flame is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation version 3.
// 
//     Flame is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
// 
//     You should have received a copy of the GNU General Public License
//     along with Flame. If not, see <http://www.gnu.org/licenses/>.

// generates a random path for saving prediction results
function randomDir() {
    var text = "flame-";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < 10; i++)
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}

// checks if a string/value is a integer 
function isInt(x) {
    return !isNaN(x) && eval(x).toString().length == parseInt(eval(x)).toString().length
}

// checks if a string/value number is a float 
function isFloat(x) {
    return !isNaN(x) && !isInt(eval(x)) && x.toString().length > 0
}

// AJAX upload function
function upload(file, temp_dir, postPredict) {
    var xhr = new XMLHttpRequest();

    // xhr.upload.addEventListener('progress', function(event) {          
    //     console.log('progess', file.name, event.loaded, event.total);
    // });

    // xhr.addEventListener('readystatechange', function(event) {
    //     console.log(
    //         'ready state', 
    //         file.name, 
    //         xhr.readyState, 
    //         xhr.readyState == 4 && xhr.status
    //     );
    // });

    xhr.ontimeout = function () {
        $("#processing").prop('hidden', true);        
        alert('File upload timed out. Results can be incomplete');
        return false;
    };

    xhr.onerror = function () {
        $("#processing").prop('hidden', true);        
        alert('File upload error!');
        return false;
    }

    xhr.onload = function () {
        postPredict(temp_dir, file.name);
    };


    xhr.open('POST', '/upload', true); 
    xhr.timeout = 600000; //600.000 ms, 10 minutes!

    xhr.setRequestHeader('X-Filename', file.name);
    xhr.setRequestHeader('Temp-Dir', temp_dir);

    xhr.send(file);

    return true;
}