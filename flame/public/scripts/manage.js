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

//TODO: Remove all development console logs and alerts.
//TODO: Show good results without console logs.


var selectModel;
function viewFullInfo() {
    var model = $("#model option:selected").text();

    $.post('/showInfo', { "model": model })


        .done(function (results) {
            console.log(results);
        });
}


/**
 * Summary. Hide all forms
 * Description. Hide the forms when you click on a form 
 * 
 */
function hideAll() {
    $("#importForm").hide("fast");
    $("#addForm").hide("fast");
}
/**
 * Summary. Display the add new model form
 * Description. First hide all forms calling hideAll and display the add new model form
 * 
 */
function displayNewModelForm() {
    $('#add').click(function () {
        hideAll();
        $("#addForm").toggle("fast");
    });
}

/**
 * Summary. Display the import model form
 * Description. First hide all forms calling hideAll and display the import model form
 * 
 */
function displayImportModelForm() {
    $('#import').click(function () {
        hideAll();
        $("#importForm").toggle("fast")
    });
}

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

/**
 * Summary. Checks if a string is alphanumeric
 * Description. Check if a string is alphanumeric using a regex
 * @param {string} name String to check 
 * 
 * @returns {boolean} true if string is alphanumeric or false otherwise
 */
function alphanumericChecker(name) {
    var letters = /^[A-Za-z0-9]+$/;
    if (name.match(letters)) {
        return true;
    }
    else {
        return false;
    }
}
/**
 * Summary. Check if a model exists
 * Description. check if a model exists calling /dir function
 * @param {string} model String to check 
 * 
 * @returns {boolean} true if model exists or false otherwise
 */
function modelExists(model) {
    console.log(model);
    $.get('/dir', function (result) {
        console.log("im in get");
        //console.log(result)
        parsedResult = JSON.parse(result);
        //console.log(parsedResult);
        for (name in parsedResult) {
            //console.log(parsedResult[name][0]+"/"+model);
            if (model === parsedResult[name][0]) {
                return true;
            }
        }
        return false;
    });
    //return false;
}
/**
 * Summary. Add a model.
 * Description. Add a model and check if it exists and if it is alphanumeric.
 * Depending on the result it shows a message or another.
 * If all is correct the combobox are reloaded to show the new element.
 */
function addModel() {
    var name = $("#name").val(); // value from input
    //console.log(modelExists(name));
    if (alphanumericChecker(name)) {   // passes check
        if (/*modelExists(name)===false*/true) {    //TODO repair modelExists func, it returns undefinded
            $.post('/addModel', { "model": name })
                .done(function (result) {
                    //console.log(result);
                }).fail(function (result) {   // !!! Say that fails but it works 
                    //alert("fail");
                });
                doneModal();
                loadTree();
        } else {
        }
    } else {
        alert("Name must be alphanumeric");
    }
}

function uploadModel() {
    var ifile = document.getElementById("uploadfile").files[0];
    if (upload(ifile, '', importModel)==false) {
        $("#data-console").text("unable to upload file, prediction aborted...");
        return;
    };
}


function importModel(temp_dir, name) {
    console.log ('importing model: '+name)

    $.post('/importModel', { "model": name })
        .always(function (result) {

        });
}

/**
 * Summary. Shows a confirm dialog
 * Description. Shows a confirm dialog with a message
 * @param {string} msg message to display 
 * @param {string} model model to display in the message 
 *
 * @returns {boolean} true if user confirms or false otherwise
 */
function confDialog(msg, model) {
    var question = msg + ": " + model + "?";
    if (confirm(question)) {
        return true;
    } else {
        return false;
    }
}

/**
 * Summary. Remove all model family
 * Description. First check if the model that the user wants to remove exists (to prevent DOM modifications), if exists 
 * the app shows a confirm dialog and if the user really wants to delete the family the family is removed.
 * Show the transaction results and updates the main tree
 */
function deleteFamily() {
    var model = $("#hiddenInput").val(); // value from input
    var modelChild = $("#hiddenInputChild").val(); // value from input
    console.log(model);
    console.log(modelExists(model));
        if (/*modelExists(model)*/true) {    //TODO repair modelExists func, it returns undefinded
            $.post('/deleteFamily', { "model": model })
                .always(function (result) {
                    console.log("pass");                // As in add it also says that it fails but it really works
                    doneModal();
                    loadTree();
                });

            console.log("Family removed");
        }
        else {
            console.log("The model doesnt exist, please update the page");

        }

}
/**
 * Summary. Remove the selected version
 * Description. Shows a confirm dialog and if the user confirm the selected model version is removed.
 * When the transaction is completed it shows the result to the user and relaod the main tree.
 */
function deleteVersion() {
    var model = $("#hiddenInput").val(); // value from input
    var modelChild = $("#hiddenInputChild").val(); // value from input
        $.post('/deleteVersion', { "model": model, "version": modelChild })
            .always(function (result) {
                console.log(result);
                doneModal();
                loadTree();
                expandNode();
            });
}

/**
 * Summary. Clone the selected model
 * Description. Shows a confirm dialog and if the user confirm the selected model is cloned
 * When the transaction is completed it shows the result to the user and relaod the main tree.
 */
function cloneModel() {
    var model = $("#hiddenInput").val(); // value from input
        $.post('/cloneModel', { "model": model })
            .always(function (result) {
                doneModal();
                loadTree();
                expandNode();
                console.log("cloned");
            });
}

/**
 * Summary. Loads the main tree
 * Description. Loads the main tree with the option provided in the constructor
 */
function loadTree() {
    $.get('/dir').done(function (result) {
        result = JSON.parse(result);
        $('#tree').treeview({
            color: undefined,
            onhoverColor: '#edba74',
            selectedBackColor: "#e59d22",
            expandIcon: 'fas fa-minus',
            collapseIcon: 'fas fa-plus',
            levels: 0,
            data: result
        });
        selectedNode();
        $("#tree ul").addClass("list-group-flush");
    });
}

/**
 * Summary. Expand all tree nodes
 */
function expandTree() {
    $("#tree").treeview("expandAll", { silent: true });
}

/**
 * Summary. Collapse all tree nodes
 */
function collapseTree() {
    $('#tree').treeview('collapseAll', { silent: true });

}
/**
 * Summary. Expand the seleted node
 */
function expandNode() {
    $('#tree').treeview('expandNode', [selectModel.nodeId, {levels: 2, silent: true} ]);
}

/**
 * Summary. Clone the selected model
 * Description. Shows a confirm dialog and if the user confirm the selected model is cloned
 * When the transaction is completed it shows the result to the user and relaod the main tree.
 */
function selectedNode() {
    console.log("Now you can select nodes")
    var query;
    $("#tree").on('nodeSelected', function (getSel, data) {
        console.log(data);
        console.log(data.text);
        parentNode = $('#tree').treeview('getParent', data);
        console.log(parentNode.text);
        $("#exportBTN").removeClass("disabled");
        $("#cloneBTN").attr("disabled", false);
        $("#deleteModelBTN").attr("disabled", false);

        // Check if the node selected is father or child
        if (typeof parentNode.text !== 'string') {     //father selected
            selectModel = data;
            console.log("father");
            //Set all texts
            $("#details").text(data.text);
            $("#manage").text(data.text);
            //Disable delete version button cuz a father is selected
            $("#deleteVersionBTN").attr("disabled", true);
            //Set hidden inputs
            $("#hiddenInputChild").val("");
            $("#hiddenInput").val(data.text);
            //Set the main table 
            $("#tBody").empty();
            $("#tBody").append("<tr><td>Select a version</td></tr>");
            $("#manage").addClass("border");
            $("#manage").addClass("rounded");
            //Sets the url to launch when the export button is pressed
            query = "exportModel?model="+data.text;
            document.getElementById("exportBTN").setAttribute("href", query); 
        } else {                                      //child selected
            selectModel = parentNode;
            console.log("child");
            //Set all text
            $("#details").text(parentNode.text + "." + data.text);
            $("#manage").text(parentNode.text + "." + data.text);
            //Enable delete version button cuz a child is selected
            $("#deleteVersionBTN").attr("disabled", false);
            //Set hidden inputs
            $("#hiddenInput").val(parentNode.text);
            $("#hiddenInputChild").val(data.text);
            //Sets the url to launch when the export button is pressed
            query = "exportModel?model="+parentNode.text;
            document.getElementById("exportBTN").setAttribute("href", query); 
            //Load the main table 
            getInfo();
            $("#manage").addClass("border");
            $("#manage").addClass("rounded");
        }
        return data;
    });
}
/**
 * Summary. Get the version details
 * Description. Get the version details and print the result in table format.
 * If the result is empty it informs the user about the error
 */
function getInfo() {
    $("#tBody").empty();
    var model = $("#hiddenInput").val();
    var version = $("#hiddenInputChild").val();
    var output = "JSON";
    $.post('/modelInfo', { "model": model, "version": version, "output": output })
        .done(function (result) {
            try {
                result = JSON.parse(result);
                var len = result.length;
                for (var i = 0; i < len; i++) {
                    $("#tBody").append("<tr class='tElement' ><td data-toggle='tooltip' data-placement='top' title='" + result[i][1] + "'>" + result[i][0] + "</td><td>" + result[i][2] + "</td></tr>");
                }
            }catch{
                $("#tBody").append("<tr><td>No info provided with this version</td></tr>");
            }
            
        });
}
/**
 * Summary: Generate the modal and show it
 * Description: Generate the modal including the text, title and function to call when the yes button is pressed
 * @param {string} title modal ttle
 * @param {string} text modal text
 * @param {string} func function to call   
 */
function generateModal(title, text, func){
    var modal = "<div class='modal fade' id='exampleModal' tabindex='-1' role='dialog' aria-labelledby='exampleModalLabel' aria-hidden='true'> \
    <div class='modal-dialog' role='document'> \
      <div class='modal-content'> \
        <div class='modal-header'> \
          <h5 class='modal-title' id='exampleModalLabel'>"+title+"</h5> \
          <button type='button' class='close' data-dismiss='modal' aria-label='Close'> \
            <span aria-hidden='true'>&times;</span> \
          </button> \
        </div> \
        <div class='modal-body' id='modalBody'> \
          "+text+" \
        </div> \
        <div class='modal-footer'> \
          <button type='button' class='btn btn-secondary' data-dismiss='modal'>Close</button> \
          <button type='button' id = 'modalYes' class='btn btn-primary' onclick='"+func+"'>Yes</button> \
        </div> \
      </div> \
    </div> \
  </div>"
  $("#modal").html(modal);
  $("#exampleModal").modal();
    $("#exampleModal").modal('show');
}
/**
 * Summary: Show a modal with a message
 * Description: Show a  modal with a message removing te yes button. By default the message is Completed
 * @param {string} msg="Completed" 
 */
function doneModal(msg="Completed") {
    $("#modalYes").remove();
    $("#modalBody").text(msg);

}



/**
 * Summary: Activates all button handlers
 * Description: Activates all button handlers and set the text modal, action and title
 */
function buttonClick() {
    $("#cloneBTN").click(function(){
        generateModal("Clone", "Do you want to clone "+$("#hiddenInput").val()+" ?", "cloneModel()");
    });
    $("#deleteVersionBTN").click(function(){
        generateModal("Remove version", "Do you want to remove "+$("#hiddenInput").val()+"."+$("#hiddenInputChild").val()+" ? <br> You will not be able to recover the version", "deleteVersion()");
    });
    $("#addE").click(function(){
        generateModal("Add", "Do you want to add "+$("#name").val()+" ? <br> A new model with the given name will be created", "addModel()");
    });
    $("#deleteModelBTN").click(function(){
        generateModal("Remove model", "Do you want to remove "+$("#hiddenInput").val()+" ? <br> You will not be able to recover the model", "deleteFamily()");
    });
}
// main
$(document).ready(function () {
    //Reset all inputs
    $("#hiddenInput").val("");
    $("#hiddenInputChild").val("");
    $("#name").val("");
    $("#importLabel").val("");
    //Disable delete version button
    $("#deleteVersionBTN").attr("disabled", true);
    $("#exportBTN").attr("disabled", true);
    $("#cloneBTN").attr("disabled", true);
    $("#cloneBTN").attr("disabled", true);
    $("#deleteModelBTN").attr("disabled", true);
    //Load the main tree
    loadTree();
    //generateTable();
    //Hide all forms
    hideAll();
    // Toggles the forms between hide and show
    displayImportModelForm();
    displayNewModelForm();
    //Activate all buttons
    buttonClick();

    $("#uploadfile").on('change',function(){
        file = document.getElementById("uploadfile").files[0];
        $("#impLabel").html( file.name ); 
        $("#predict").prop('disabled', false);
    })
    
    

});