//     Description    Flame Predict JavaScript 
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


// sort keys to order columns in a logical way
function sortKeys (myjson) {
   
    // select keys and order logically
    //  1. obj_name
    //  2. SMILES (if present at all)
    //  3. main results (one or many)
    
    // 1. obj_name
    var key_list = ['obj_nam'];
    
    // 2. SMILES
    if (myjson.hasOwnProperty('SMILES')){
        key_list.push('SMILES');
    }
    
    // 3. main results
    var main = myjson['meta']['main'];
    key_list = key_list.concat(main);
    
    // special keys, already processed (obj_nam, SMILES and main[])
    // or never shown in a table (origin and meta)
    const black_list = key_list.concat(['origin','meta']);

    for (var key in myjson){
        if ( ! black_list.includes(key)){
            key_list.push(key);
        }
    }

    return key_list;
}


// parse results obtained from the prediction
function parseResults (results) {
    $("#data-body").text(results);

    lastResults = results;
    
    var myjson = JSON.parse(results);
    var mainv = myjson['meta']['main'][0];

    key_list = sortKeys(myjson);
    
    // header
    var tbl_body = '<thead><tr><th>#</th>';
    for (var key in key_list){
        label = key_list[key];
        label = label.replace (/_/g , " ");
        tbl_body +=  '<th>'+label+'</th>';
    }
    
    // body
    tbl_body+='</tr></thead>';
    var val;
    var val_float;

    for (var i in myjson[mainv]){

        tbl_body += "<tr><td>"+(+i+1);

        for (var key in key_list){

            if (key_list[key]=="SMILES"){
                tbl_body += '<td><canvas id="mol'+i+'"></canvas>';
                continue;
            } 

            val = myjson[key_list[key]][i];

            if (val==null) {
                tbl_body += "</td><td> - ";
                continue;
            }

            val_float = parseFloat(val);

            if(isNaN(val_float)){
                tbl_body += "</td><td>"+val;
                continue;
            }

            tbl_body += "</td><td>"+val_float.toFixed(3);

        }
        tbl_body += "</td></tr>";
    }
    
    $("#data-table").html(tbl_body);   

    // SMILES must be inserted after the canvases were already created in included in the HTML code
    if (key_list.includes('SMILES')){

        let smilesDrawer = new SmilesDrawer.Drawer( {'width':300, 'height':150});

        for (var i in myjson[mainv]){
            SmilesDrawer.parse(myjson['SMILES'][i], function(tree) {
                smilesDrawer.draw(tree, 'mol'+i, 'light', false);
            });
        }
    }

    // now we can export the results
    $("#export").prop('disabled', false);
    $("#processing").prop('hidden', true);
};


// POST a prediction request for the selected model, version and input file
function postPredict (temp_dir, ifile) {
    // collect all data for the post and insert into postData object
    
    var version = $("#version option:selected").text();
    if (version=='dev') {
        version = '0';
    };

    $.post('/predict', {"ifile"   : ifile,
                        "model"   : $("#model option:selected").text(),
                        "version" : version,
                        "temp_dir": temp_dir
                        })
    .done(function(results) {
        lastResults = results;
        parseResults (results)
    });

};


// simple utility function to download as a file text generated here (client-side)
function download(filename, text) {
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
  
    element.style.display = 'none';
    document.body.appendChild(element);
  
    element.click();
  
    document.body.removeChild(element);
}  


// main
$(document).ready(function() {

    // no prediction so far
    lastResults = null;
    
    // initialize button status to disabled on reload
    $("#predict").prop('disabled', true);
    $("#export").prop('disabled', true);
    
    
    // show file value after file select 
    $("#ifile").on('change',function(){
        file = document.getElementById("ifile").files[0];
        $("#ifile-label").html( file.name ); 
        $("#predict").prop('disabled', false);
    })
    
    var versions; // object where model name and versions are stored
    
    // ask the server about available models and versions
    $.get('/dir')
    .done(function(results) {
        
        versions = JSON.parse(results);
        
        // set model selector
        var model_select = $("#model")[0];
        for (vi in versions) {
            imodel = versions[vi][0];
            model_select.options[vi] = new Option(imodel, +vi+1)
        }

        // set version selector
        var var_select = $("#version")[0];
        vmodel = versions[0][1];
        for (vj in vmodel) {
            var_select.options[vj] = new Option(vmodel[vj],+vj+1);
        }

    });

    // define available versions for this endpoint
    $("#model").on('change', function (e) {
        $("#version").empty();
        var var_select = $("#version")[0];
        for (vi in versions) {
            if (versions[vi][0] == $("#model option:selected").text()){

                for (vj in versions[vi][1]) {
                    var_select.options[vj] = new Option(vmodel[vj],+vj+1);
                }
                return;
            }
        }
    });

    // "predict" button
    $("#predict").click(function(e) {

        // make sure the browser can upload XMLHTTP requests
        if (!window.XMLHttpRequest) {
            $("#data-body").text("this browser does not support file upload");
            return;
        };

        $("#processing").prop('hidden', false);

        // clear GUI
        $("#data-body").text('processing... please wait');
        $("#data-table").html('');

        $("#export").prop('disabled', true);
         
        // get the file 
        var ifile = document.getElementById("ifile").files[0];
        
        // generate a random dir name
        var temp_dir = randomDir();

        // call postPredict when file upload is completed
        if (upload(ifile, temp_dir, postPredict)==false) {
            $("#data-body").text("unable to upload file, prediction aborted...");
            return;
        };

        e.preventDefault(); // from predict click function
    });

    $("#export").click(function(e) {

        if (lastResults==null)
            return;

        var myjson = JSON.parse(lastResults);
        var mainv = myjson['meta']['main'][0];

        key_list = sortKeys(myjson);

        var tsv='';

        // header
        for (var key in key_list){
            label = key_list[key];
            label = label.replace (/_/g , " ");
            tsv +=  label+'\t';
        }
        tsv += '\n';

    
        // body
        var val;
        var val_float;
        for (var i in myjson[mainv]){
            for (var key in key_list ){
                val = myjson[key_list[key]][i];
                if (val==null) {
                    tsv +=  " - \t";
                }
                else {
                    val_float = parseFloat(val);
                    if(isNaN(val_float)){
                        tsv +=  val.toString()+'\t';
                    }
                    else {
                        var vtemp = val_float.toFixed(3);
                        tsv +=  vtemp.toString()+'\t';
                    }
                }
            }
            tsv += "\n";
        }

        download("results.tsv",tsv);

        e.preventDefault(); // from predict click function
    });


});