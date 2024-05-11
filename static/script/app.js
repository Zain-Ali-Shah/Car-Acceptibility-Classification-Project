let classes
function loadData(){
    // Get the file input element
    var inputFile = document.getElementById('csvFile')
    // Get the selected file
    var selectedFile = inputFile.files[0]
    var file_nam=selectedFile['name']
    console.log(file_nam)
    var file_name={
      'file_name' : file_nam
    }
    fetch('http://localhost:5000/load_data', {
        method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(file_name),
      })
      .then(response => response.json())
      .then(data => {
        var resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';
        var messageElement = document.createElement('h1')
        messageElement.innerText = data['message']
        resultContainer.appendChild(messageElement)
        if (data.hasOwnProperty('class_names')){
          classes=data['class_names']
          document.getElementById('myButton2').removeAttribute('disabled');
        }else{
          for(var i=2;i<=9;i++){
            document.getElementById('myButton' + i).disabled=true;
          }
        }
      })
      .catch(error => {
        // Handle errors
        console.error('Error:', error);
      });
}

function prepareData(){
    fetch('http://localhost:5000/prepare_data', {
        method: 'GET',
      })
      .then(response => response.json())
      .then(data => {
        var resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';
        var messageElement = document.createElement('h1');
        messageElement.innerText = data['message']
        resultContainer.appendChild(messageElement);
        console.log(data);
        document.getElementById('myButton3').removeAttribute('disabled');
      })
      .catch(error => {
        console.error('Error:', error);
      });
}

function setActive(id,classs){
  var dropdownItems = document.querySelectorAll(classs);
  dropdownItems.forEach(function(item) {
    item.classList.remove('active');
  });
  document.getElementById(id).classList.add('active');
}

function trainModel(dropdownId){
    var dropdownMenu = document.getElementById(dropdownId);
    var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
    selectedModel=selectedDropdownItem.innerText;
    
    path='';
    if (selectedModel=="Logistic Regression"){
      path='http://localhost:5000' + '/logistic_regression'
    }
    else if (selectedModel=="Decision Tree"){
      path='http://localhost:5000' + '/decision_tree'
    }
    else if (selectedModel=="Random Forest"){
      path='http://localhost:5000' + '/random_forest'
    }
    else if (selectedModel=="K Nearest Neighbors"){
      path='http://localhost:5000' + '/knn'
    }
    else {
      path='http://localhost:5000' + '/svm'
    }
    fetch(path, {
        method: 'GET',
      })
      .then(response => response.json())
      .then(data => {
        var resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';
        var messageElement = document.createElement('h1');
        messageElement.innerText = data['message']
        resultContainer.appendChild(messageElement);
        console.log(data);
        for(var i=4;i<=9;i++){
          document.getElementById('myButton' + i).removeAttribute('disabled');
        }
      })
      .catch(error => {
        console.error('Error:', error);
      });
}

function calculateConfusionMatrix(dropdownId){
    var dropdownMenu = document.getElementById(dropdownId);
    var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
    selectedModel=selectedDropdownItem.innerText;
    console.log(selectedModel)
    var requestData = {
      'model_name': selectedModel,
    };
    fetch('http://localhost:5000/cal_confusion_matrix', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    })
    .then(response => response.json())
    .then(data => {
      var resultContainer = document.getElementById('resultContainer');
      resultContainer.innerHTML = '';
      var messageElement = document.createElement('h1');
      messageElement.innerText = 'Confusion Matrix for ' + selectedModel;
      resultContainer.appendChild(messageElement);
      // Display the confusion matrix as a table
      var confusionMatrixTable = document.createElement('table');
      confusionMatrixTable.classList.add('table', 'table-bordered', 'table-striped', 'mt-3', 'bg-info', 'text-white');
      // Add headers
      var headerRow = confusionMatrixTable.insertRow();
      headerRow.insertCell().innerText = 'Actual \\ Predicted';
      for (var i = 0; i < 4; i++) {
        var headerCell = headerRow.insertCell();
        headerCell.innerText = classes[i];
      }
      // Add data rows
      confusionMatrix=data['Confusion Matrix']
      for (var i = 0; i < confusionMatrix.length; i++) {
        var dataRow = confusionMatrixTable.insertRow();
        dataRow.insertCell().innerText = classes[i];
        for (var j = 0; j < confusionMatrix[i].length; j++) {
          var dataCell = dataRow.insertCell();
          dataCell.innerText = confusionMatrix[i][j];
        }
      }
      resultContainer.appendChild(confusionMatrixTable);
      console.log(data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

function calculateSensitivity(dropdownId){
  var dropdownMenu = document.getElementById(dropdownId);
  var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
  selectedModel=selectedDropdownItem.innerText;
  console.log(selectedModel)
  var requestData = {
    'model_name': selectedModel,
  };
  fetch('http://localhost:5000/cal_sensitivity', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
  .then(response => response.json())
  .then(data => {
    var sensitivityValues=data['Sensitivity']
    var resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = '';
    var messageElement = document.createElement('h1');
    messageElement.innerText = 'Sensitivity of ' + selectedModel;
    resultContainer.appendChild(messageElement);
    // Display the sensitivity values in a table
    var sensitivityTable = document.createElement('table');
    sensitivityTable.classList.add('table', 'table-bordered', 'mt-3', 'bg-info', 'text-white');
    // Add headers
    var headerRow = sensitivityTable.insertRow();
    for (var i = 0; i < 4; i++) {
      var headerCell = headerRow.insertCell();
      headerCell.innerText = classes[i];
    }
    // Add a single row
    var sensitivityRow = sensitivityTable.insertRow();
    for (var i = 0; i < sensitivityValues.length; i++) {
      var sensitivityCell = sensitivityRow.insertCell();
      sensitivityCell.innerText = sensitivityValues[i].toFixed(2);
    }
    resultContainer.appendChild(sensitivityTable);
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function calculateSpecificity(dropdownId){
  var dropdownMenu = document.getElementById(dropdownId);
  var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
  selectedModel=selectedDropdownItem.innerText;
  console.log(selectedModel)
  var requestData = {
    'model_name': selectedModel,
  };
  fetch('http://localhost:5000/cal_specificity', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
  .then(response => response.json())
  .then(data => {
    specificityValues=data['Specificity']
    var resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = '';
    var messageElement = document.createElement('h1');
    messageElement.innerText = 'Specificity of ' + selectedModel;
    resultContainer.appendChild(messageElement);
    // Display the specificity values in a table
    var specificityTable = document.createElement('table');
    specificityTable.classList.add('table', 'table-bordered', 'mt-3', 'bg-info', 'text-white');
    // Add headers
    var headerRow = specificityTable.insertRow();
    for (var i = 0; i < 4; i++) {
      var headerCell = headerRow.insertCell();
      headerCell.innerText = classes[i];
    }
    // Add a single row
    var specificityRow = specificityTable.insertRow();
    for (var i = 0; i < specificityValues.length; i++) {
      var specificityCell = specificityRow.insertCell();
      specificityCell.innerText = specificityValues[i].toFixed(2);
    }
    resultContainer.appendChild(specificityTable);
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function plotROC(dropdownId){
  var dropdownMenu = document.getElementById(dropdownId);
  var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
  selectedModel=selectedDropdownItem.innerText;
  console.log(selectedModel)
  var requestData = {
    'model_name': selectedModel,
  };
  fetch('http://localhost:5000/roc_curve', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
  .then(response => response.json())
  .then(data => {
    imageUrl=data['ImageURL']
    var resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = '';
    var messageElement = document.createElement('h1');
    messageElement.innerText = 'ROC curve for ' + selectedModel;
    resultContainer.appendChild(messageElement);
    var resultImageElement = document.createElement('img');
    resultImageElement.src = imageUrl;
    resultImageElement.alt = 'Result Image';
    resultImageElement.classList.add('img-fluid', 'mt-3');
    resultContainer.appendChild(resultImageElement);
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function calculateAUC(dropdownId){
  var dropdownMenu = document.getElementById(dropdownId);
  var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
  selectedModel=selectedDropdownItem.innerText;
  console.log(selectedModel)
  var requestData = {
    'model_name': selectedModel,
  };
  fetch('http://localhost:5000/auc', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
  .then(response => response.json())
  .then(data => {
    auc=data['AUC']
    var resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = '';
    var messageElement = document.createElement('h1');
    messageElement.innerText = 'Area Under Curve for ' + selectedModel;
    resultContainer.appendChild(messageElement);
    // Display the values in a table
    var aucTable = document.createElement('table');
    aucTable.classList.add('table', 'table-bordered', 'mt-3', 'bg-info', 'text-white');
    // Add headers
    var headerRow = aucTable.insertRow();
    for (var i = 0; i < 4; i++) {
      var headerCell = headerRow.insertCell();
      headerCell.innerText = classes[i];
    }
    // Add a single row
    var aucRow = aucTable.insertRow();
    for (var i = 0; i < auc.length; i++) {
      var aucCell = aucRow.insertCell();
      aucCell.innerText = auc[i].toFixed(2);
    }
    resultContainer.appendChild(aucTable);
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function senAtSpec90(dropdownId){
  var dropdownMenu = document.getElementById(dropdownId);
  var selectedDropdownItem = dropdownMenu.querySelector('.dropdown-item.active');
  selectedModel=selectedDropdownItem.innerText;
  console.log(selectedModel)
  var requestData = {
    'model_name': selectedModel,
  };
  fetch('http://localhost:5000/sen_at_spec_90', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
  .then(response => response.json())
  .then(data => {
    senSpec90=data['Sensitivity at specificity = 0.90']
    var resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = '';
    var messageElement = document.createElement('h1');
    messageElement.innerText = 'Sensitivity at specificity = 0.90 of ' + selectedModel;
    resultContainer.appendChild(messageElement);
    // Display the values in a table
    var senSpec90Table = document.createElement('table');
    senSpec90Table.classList.add('table', 'table-bordered', 'mt-3', 'bg-info', 'text-white');
    // Add headers
    var headerRow = senSpec90Table.insertRow();
    for (var i = 0; i < 4; i++) {
      var headerCell = headerRow.insertCell();
      headerCell.innerText = classes[i];
    }
    // Add a single row
    var senSpec90Row = senSpec90Table.insertRow();
    for (var i = 0; i < senSpec90.length; i++) {
      var senSpec90Cell = senSpec90Row.insertCell();
      senSpec90Cell.innerText = senSpec90[i].toFixed(2);
    }
    resultContainer.appendChild(senSpec90Table);
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}
