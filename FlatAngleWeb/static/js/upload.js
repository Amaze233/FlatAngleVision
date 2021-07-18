const dropArea = document.querySelector(".drag_area");
const dragText = dropArea.querySelector("header");
const button = dropArea.querySelector("button");
const input1 = document.getElementById("inputfile");
const input2 = document.getElementById("uploadfile");
const img = dropArea.querySelector("img");
const buttonArea = document.querySelector('.button_area');
// const buttonArea = document.getElementById("button_area");
const uploadButton = buttonArea.querySelector("button");
// const input2 = buttonArea.querySelector('input');

const inputFile = document.getElementById('inputfile');

let file;//文件
let filePath;
let fileURL;
let fileReader;

//获取地址
inputFile.onchange = function () 
{
	let url = null;
	let fileObj = document.getElementById("inputfile").files[0];
	if (window.createObjcectURL != undefined) {
		url = window.createOjcectURL(fileObj);
	} else if (window.URL != undefined) {
		url = window.URL.createObjectURL(fileObj);
	} else if (window.webkitURL != undefined) {
		url = window.webkitURL.createObjectURL(fileObj);
	}
	console.log(url)
}

//点击按钮调用隐藏的input的点击效果	
button.onclick = () =>
{
	input1.click();
}

uploadButton.onclick = () =>
{
	input2.click();
	fileReader.addEventListener("load", function (ev) {
		// 文件读取成功后调用此方法。
		// 通过 fileReader.result 即可获取到文件内容。
		var result = fileReader.result;
		$.post("https://www.microanswer.cn/test/uploadBase64", {
			base64Data: result
		}, function (response) {
			// 服务器响应了我们的上传请求。
		});
	});
}


function UploadFile()
{
	console.log(file);
	uploadButton.style.display = "none";
	buttonArea.classList.add("active");
}

// uploadButton.onclick = () =>
// {
// 	// input2.click();
// 	UploadFile();
// }


//若用户将图片选定
//加事件监听器
input1.addEventListener("change", function(){
	file = this.files[0];
	dropArea.classList.add("active");
	showFile();
	popButton();
	dropArea.classList.add("showFile");
})

//若用户将图片拖拽至选定区域内
//加事件监听器
dropArea.addEventListener("dragover",(event)=>{
	event.preventDefault();
	// console.log("File is over DragArea");
	dropArea.classList.add("active");
	//dropArea.style.backgroundColor = #ddd;
	// function change()
	// {
	// 	dropArea.style.backgroundColor = #ddd;
	// }
	dragText.textContent = "Release to Upload File";
})

//若用户将图片拖离选定区域内
//加事件监听器
dropArea.addEventListener("dragleave",()=>{
	// console.log("File is outside from DragArea");
	dropArea.classList.remove("active");
	dragText.textContent = "Drag & Drop To Upload File";
})

//若用户将图片拖放至选定区域内
//加事件监听器
dropArea.addEventListener("drop",(event)=>{
	event.preventDefault();
	// console.log("File is dropped on DragArea");
	// 获取用户选择文件和[0]这意味着如果用户选择多个文件，
	// 那么我们只选择第一个
	file = event.dataTransfer.files[0];
	showFile();
	popButton();
	dropArea.classList.add("showFile");
})

//图片预览
function showFile()
{
	let fileType = file.type;
	console.log(fileType);
	
	
	
	
	// var src = event.target || window.event.srcElement; //获取事件源，兼容chrome/IE
	// filePath = src.value;//图片完整路径
	// filePath = getFilePath(input1);
	// console.log("this is me");
	
	//吐了
	// var url;
	// var name = document.getElementById("file").files[0];
	// if (window.URL != undefined && name) {
	// 	url = window.URL.createObjectURL(name);
	// }
	// console.log(document.getElementById("file").value);
	// console.log(url);
	
	// console.log(filePath);
	
	// let vaildExtensions = ["image/png","image/jpg","image/jpeg"];
	
	
	
	if (!fileType.startsWith("image/")) {
		alert("输入的不是图片！");
		dropArea.classList.remove("active");
	} else{
		console.log("This is an image!");
		//创建图片预览所需的FileReader对象
		fileReader = new FileReader();
		// console.log(fileReader.readAsDataURL(file));
		fileReader.onload = ()=>{
			//传入URL
			fileURL = fileReader.result;

			console.log(fileURL);
			let imgTag = `<img src = "${fileURL}"  alt = "">`;
			//图片预览
			dropArea.innerHTML = imgTag;
		}
		fileReader.readAsDataURL(file);
		dropArea.classList.add("showFile");
	}
}

//获取真实地址
function getFilePath(input){
	if(input){//input是<input type="file">Dom对象
		if(window.navigator.userAgent.indexOf("MSIE")>=1){  //如果是IE  
	        input.select();    
	      return document.selection.createRange().text;    
	    }    
	    else if(window.navigator.userAgent.indexOf("Firefox")>=1){  //如果是火狐  {    
	        if(input.files){    
	        	return input.files.item(0).getAsDataURL();    
	        }    
	      	return input.value;    
	    }    
	    return input.value; 
	}
}


//弹入按钮
function popButton()
{
	buttonArea.style.display = "block";
	
	// let button1 = `<button type="button"> Browse File </button>`;
	// buttonArea.innerHTML = button1;
	// buttonArea.innerHTML = "<br/>";

	// buttonArea.append(button1);
	// buttonArea.append(button2);
	
	// var button1 = buttonArea.createElement("button");
	// input.setAttribute('type', 'button'); 
	// button1.innerText("Browse File");
	// buttonArea.appendChild(button1);
	
	// var br = buttonArea.createElement("br");
	// buttonArea.appendChild(br);
	
	// var button2 = buttonArea.createElement("button");
	// input.setAttribute('type', 'button'); 
	// button2.innerText("Browse File");
	// buttonArea.appendChild(button2);
	
}