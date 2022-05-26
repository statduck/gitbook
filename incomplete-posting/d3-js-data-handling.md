# D3-js\[Data handling]

## Local server hosting

1. Open a new terminal in Mac
2. cd \~/Desktop/project-folder
3. python -m SimpleHTTPServer 8888 &.filename

## Chaining

```javascript
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset = "utf-8">
        <title> D3 Page Template</title>
        <script type="text/javascript" src="d3.js"></script>
    </head>
    <body>
        <script type="text/javascript">
        d3.select("body").append("p").text("New paragraph!");
        </script>
    </body>
</html>

```

* d3: Refer the D3 object.
* .select("body"): Give the select() method a CSS selector as input
* append("p"): empty p paragraph is appended to the body.
* .text("New paragraph!")

Handoff: D3 methods return a **selection** (So it uses chaining)

Without chaining we can also re**-**paragraph above code.

```javascript
var body = d3.select("body");
var p = body.append("p");
p.text("New paragraph!");
```

## Binding data

```javascript
d3.csv("food.csv", function(data) {
    console.log(data);
});
```

csv(): the path, call back function

When the call back function is called, the csv file is already imported. Every value in csv is stored as a string, even the number. So converting is needed. While the csv fild is loaded, the rest of a code is executed so assigning global variable would be helpful.

{% tabs %}
{% tab title="JavaScript" %}
```javascript
var rowConverter = function(d) {
    return {
        Food: d.Food, //No conversion
        Deliciousness: parseFloat(d.Deliciousness)
    };
}

d3.csv("food.csv", rowConverter, function(data) {
    console.log(data);
});
```
{% endtab %}

{% tab title="Global var" %}
```javascript
var dataset;

d3.csv("food.csv", function(data) {
    dataset = data; // Once loaded, copy to dataset.
    generateVis(); // Then call other functions that
    hideLoadingMsg(); // depend on data being present.
});

var useTheDataLater = function() {
    //Assuming useTheDataLater() is called sometime after
    //d3.csv() has successfully loaded in the data,
    //then the global dataset would be accessible here.
};
```
{% endtab %}

{% tab title="Global var2" %}
```javascript
var dataset; //Declare global var

d3.csv("food.csv", function(data){
    // Hand CSV data off to global var,
    // so it's accessbile later.
    
    dataset = data;
    
    //Call some other functions that
    //generate your visualization, e.g.:
    generateVisualization();
    makeAwesomeCharts();
    makeEvenAwesomerCharts();
    thankAwardsCommittee();
});
```
{% endtab %}

{% tab title="Handling error" %}
```javascript
var dataset;

d3.csv("food.csv", function(error,data){
    if (error){ //If error is not null, st went wrong.
        console.log(error); //Log the error.
    } else { //If no error, the file loaded correctly.
        console.log(data); // Log the data.
        //Include other code to execute after successful file load here
        dataset = data;
        generateVis();
        hideLoadingMsg();
    }
});
    
```
{% endtab %}
{% endtabs %}



## Selection

```javascript
var dataset = [5,10,15,20,25];

d3.select("body").selectAll("p")
    .data(dataset)
    .enter()
    .append("p")
    .text("New paragraph!")
    //.text(function(d) { return d; });
```

We have to select elements that don't yet exist. The magical method is "enter"

* d3.select("body"): Finds the body in the DOM
* .selectAll("p"): Selects all paragraphs in the DOM. This returns an empty selection because none exist yet.
* .data(dataset): Counts and parses our data values.
* enter(): Creates a new placeholder element
* .append("p"): appends a p element
* .text("New paragraph!"): inserts a text value into p

If we use **.text(function(d) { return d; });** it populates the contents of each paragraph from our data.

## Functioning

{% tabs %}
{% tab title="Anonymous function" %}
```javascript
function(input_value) {
    //Calculate something here
    return output_value;
}
```
{% endtab %}

{% tab title="Named function" %}
```javascript
var doSomething = function() {
    //Code to do something here
};
```
{% endtab %}

{% tab title="Wrapped by text" %}
```javascript
.text(function(d) { // <- Note tender embrace at left
    return "I can cout up to " + d;
});

// Our anonymous function is executed first.
```
{% endtab %}

{% tab title="Style" %}
```javascript
.style("color", function(d) {
    if (d > 15) { //Threshold of 15
        return "red";
    } else {
        return "black" ;
    }
});
```
{% endtab %}
{% endtabs %}



## Div and Attr

{% tabs %}
{% tab title="div(1)" %}
```javascript
var dataset = [5,10,15,20,25];
<div style = "display: inline-block;
                width: 20px;
                height: 75px;
                background-color: teal;"></div>
```
{% endtab %}

{% tab title="div(2)" %}
```javascript
div.bar{
    display: inline-block;
    width: 20px;
    height: 75px; /* We'll override height later */
    background-color: teal;
}

<div class="bar"></div>
```
{% endtab %}

{% tab title="attr(1)" %}
```javascript
<p class="caption">
<select id="country">
<img src="logo.png" width="100px" alt="Logo" />
```
{% endtab %}
{% endtabs %}

div needs to be assigned the bar class. To add a class to an element, we use the attr() method. (not style()) style() applies CSS styles directly to an element.

| Attribute | Value    |
| --------- | -------- |
| class     | caption  |
| id        | country  |
| src       | logo.png |
| width     | 100px    |
| alt       | Logo     |

To assign a class of bar we can use: .attr("class","bar")

## Classes

Element's class is stored as an HTML attribute.

{% tabs %}
{% tab title="class" %}
```javascript
.classed("bar", true)
// If true->false, it removesthe class of bar.
```
{% endtab %}

{% tab title="bars" %}
```javascript
var dataset = [5,10,15,20,25];

d3.select("body").selectAll("div")
    .data(dataset)
    .enter()
    .append("div")
    .attr("class","bar");


```
{% endtab %}
{% endtabs %}

