const previewButton = document.querySelector(".preview_button");
const uploadButton = document.querySelector(".upload_button");
const button1 = document.getElementById("input");
const button2 = document.getElementById("upload");
const button3 = document.getElementById("select");

//点击按钮previewButton调用隐藏的button1的点击效果
previewButton.onclick = () =>
{
    button1.click();

}

//点击按钮uploadButton调用隐藏的button2的点击效果
function inputFile()
{
    button2.click();
    // alert("hello");
    button3.click();
    // alert("why?");
}

// //点击按钮uploadButton调用隐藏的button2的点击效果
// uploadButton.onclick = () =>
// {
//     alert("hello!");
//     button2.click();
// }