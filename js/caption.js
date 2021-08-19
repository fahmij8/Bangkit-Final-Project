$('#image').hide();
$('#btn-spinner').hide();
$('#spinner-caption span').hide();

let img = document.querySelector("#image");
let button = document.querySelector("#btn");
let text = document.querySelector("#txt");
let capField = document.querySelector("#caption");
let imageLoader = document.querySelector("#imageLoader");


let isModelLoaded = false;

let model;
let mobileNet;

const maxLen = 34; // 40

function preprocess(imgElement) {
    $('#spinner-caption span').show();
    $('#btn').attr('disabled',true);
    

    return tf.tidy(() => {
        let tensor = tf.fromPixels(imgElement).toFloat();
        const resized = tf.image.resizeBilinear(tensor,[224,224]);
        const offset = 127.5;
        const normalized = resized.div(offset).sub(tf.scalar(1.0));
        const batched = normalized.expandDims(0);
        return batched;
    });
}


function caption(img) {
    
    // should use promise and async-await to make it non blocking
    // max_len change karna
    let flattenLayer = tf.layers.flatten();
    console.log("Inside caption()");
    
    return tf.tidy(()=> {

        let startWord = ['<start>'];
        let idx,wordPred;
        let e = flattenLayer.apply(mobileNet.predict(img));
        
        while (true) {
            let parCaps = [];
            for (let j = 0; j < startWord.length; ++j) {
                //console.log(typeof(parCaps));
                parCaps.push(word2index[startWord[j]]);
            }
            parCaps = tf.tensor1d(parCaps)
                        .pad([[0, maxLen - startWord.length]])
                        .expandDims(0);

            
            let preds = model.predict([e,parCaps]);
            preds = preds.reshape([preds.shape[1]]);
            //console.log(preds.shape);
            
            idx = preds.argMax().dataSync();
            wordPred = index2word[idx];
            
            startWord.push(wordPred);            
            if(wordPred=='<end>'||startWord.length>maxLen)
                break;
        }
        
        // removing first and last tokern <start> and <end>
        startWord.shift();
        startWord.pop();
        
        cap = startWord.join(' ');
        console.log("caption: ", cap);
        capField.innerHTML = cap;
        $('#btnPlay').show();
		captionSpeech = cap
        playSentence(captionSpeech);
        //return startWord.join(' ');
        //return asyncCaption(img);

        $('#btn').prop('disabled',false);
        $('#btn').text('Generate Caption');
        $('#spinner-caption span').hide();
    }); 
}


async function start() {    
    //mobileNet = loadMobileNet();
    const mobilenet = await tf.loadModel('img_model_mobilenet/model.json');
    const layer = mobilenet.getLayer('conv_preds');
    console.log("mobileNet loaded");
    mobileNet =  tf.model({
        'inputs': mobilenet.inputs,
        'outputs': layer.output
    });

    model = await tf.loadModel('lang_model_lstmbidi/model.json');
    console.log("Inside start()");
    $('#spinner').hide();
    modelLoaded();
    //return true;
}

function modelLoaded() {
    mobileNet.predict(tf.zeros([1, 224, 224, 3])).dispose();
    //console.log("Inside modelLoaded()");
    isModelLoaded = true;
    text.innerHTML = "Good to go!";
}


imageLoader.addEventListener('change', function(e) {
    let reader = new FileReader();
    reader.onload = function() {
        img.src = reader.result;
        $('#image').show();
    }
    reader.readAsDataURL(e.target.files[0]); 
}, false);

function disableButton(){
    $('#spinner-caption').show();
    $('#btn').attr('disabled',true);
    $('#btn').text('Generating caption..');
    $('#btnPlay').show();
}

button.addEventListener("click",function() {
    $('#spinner-caption span').show();
    $('#btn').text('Generating Caption..');
    $('#btn').attr('disabled',true);
    $('#btnPlay').hide();
    capField.innerHTML = "";

    setTimeout(function(){
        //console.log("button pressed");
    if(!isModelLoaded) {
        $('#alerting').toast('show');
        console.log('Models not loaded yet');
        return;
    }
    
    capField.innerHTML = "Generating Caption ... Please Wait";
    let picture = preprocess(img);
    //let cap = caption(picture);
    caption(picture);
    //console.log("caption: "+ cap);
    //capField.innerHTML = cap;

    },3000);
   
    
});

function playSentence(text) {    
      let msg = new SpeechSynthesisUtterance(text);
      msg.lang = 'en-US';
      speechSynthesis.speak(msg);
}

$("#btnPlay").click(function() {
      playSentence(captionSpeech);
});

let checkGPU = () => {
    $('#image').hide();
    $('#btnPlay').hide();
    let canvas = document.getElementById("webgl-canvas")
    gl = canvas.getContext("experimental-webgl");
    let maxTextureSize = gl.getParameter( gl.MAX_TEXTURE_SIZE );

    if (maxTextureSize > 7999){
        start();
        console.log("Device is good");
    } else {
        $('#alerting2').toast('show');
        console.log("Device unable to do prediction")
    }
}

checkGPU()
