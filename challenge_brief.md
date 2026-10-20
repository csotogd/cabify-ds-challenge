## Data Science Challenge

The purpose of this challenge is to assist us in evaluating candidates for a role in our Data Science team. We only pass this challenge to candidates that we feel have a solid background and could be a good fit for our team. We appreciate you taking this time to help ensure we are a good match for each other.

### Tips
- Include code, graphics and text in a combined output. Tell a story, and let us understand as clearly as possible your thought and analytical process.

### Part 1: Experiment design
#### Background

JustEat Limited is a delivery service founded in 2001 in Kolding, Denmark. It acts as an intermediary between independent takeaway food outlets and customers. The platform enables customers to search for local takeaway restaurants, place orders, pay online and to choose from pick-up or delivery options.

As part of its value proposition, JustEat launched a free photography service where professionals would go to restaurants and take high-quality photos of the food.  From their accounts, restaurants would be able to learn more details about the service, request a professional photographer, and subsequently (after the photo shoot) have their restaurant listing updated with the professional photos. This had the goal of making the food more attractive to customers and increasing the likelihood of them making an order.

The project initially proved to be a success:
- Customers were more likely to order food that had professional photographs.
- Restaurants were able to charge more for food with professional photos.

However, over time this also became a multimillion-dollar operation and a challenge to manage across hundreds of cities. 

Fast forward to 2023: because of COVID19, customers are more used to ordering food online than before. Also, there are now new, easily available resources like high-quality smartphone cameras or specialized photography online courses that help restaurants to take good quality photos on their own. 

#### Challenge

Since the professional photography service consumes so many operational and financial resources, it is unclear whether keeping the service active is worth the cost. JustEat management have asked the Data Science team to analyse the impact of the professional photography service in order to determine whether or not they should continue funding the service. 

1. Provide full details about how you will run experiments to assess the impact of this service on both restaurants and customers. How will you ensure that the experiments are valid and not biased? 

## Part 2: Model Prototyping

### Background

Cabify serves hundreds of thousands of journeys every day. When a journey is requested, but before it takes place, we need to set a price for it. Pricing in Cabify is calculated based on the routes’ distance and duration estimated by our routing engine. To guarantee that we are offering a fair price, we must make sure that these estimates are accurate.

### Proposal

Cabify’s Data Science team understands that in this context, “accurate” means that the estimated and real routes are very similar (the driver followed the route we predicted). We can use humans to label which routes are similar and which routes are different, as it is easy for humans to look at a map of the two routes and understand what is happening. But Cabify serves millions of journeys, so the team is considering setting up a system to do it automatically.

We want to build a model that, given a pair of routes, assesses whether they are different or not. As a preliminary step, we ran a labeling project to provide the ground truth for a sample of routes on which the model can be trained. The steps were the following:

* 5,000 journeys were randomly selected.
* For each journey, estimated and real routes were plotted on a map.
* Several people (annotators) visually assess, according to their own perception, if the routes are equal, different or if they are not able to tell (annotations).

Below you can see various examples of route evaluated estimations. The magenta route is the real route while black route is the estimated route.

![example_1](https://i.imgur.com/gBVdzQX.png)
![example_2](https://i.imgur.com/K3eC0jO.png)
![example_3](https://i.imgur.com/8S58PiO.png)

### Data description

With the labeling project we obtained [a json file](https://www.dropbox.com/s/rdm11r99pmi13if/challenge_dataset.json?dl=1) where each object represents an annotator's assessment of the similarity of an estimated/real routes pair, with the following attributes:

* journey_id: an identifier for each journey.
* annotator: an identifier for each evaluator.
* annotation: selected label for the estimated/actual route pair (“Both are the same”, “They differ” and “I don’t know”).
* estimated_route: coordinates pairs (lat-lon) of the estimated route.
* real_route: coordinates pairs (lat-lon) of the real route.

Example:

```
{
    "journey_id": "9cd6cd52-54c5-11ec-ae0a-0d030544d074",
    "annotator": 3,
    "annotation": "Both are the same",
    "estimated_route": [
        [
            -12.11149,
            -77.04675
        ],
        [
            -12.11145,
            -77.04671
        ],
        [
            -12.11139,
            -77.04664
        ],
        ...
    ],
    "real_route": [
        [
            -12.111973,
            -77.047275
        ],
        [
            -12.111933,
            -77.047236
        ],
        [
            -12.111811,
            -77.047122
        ],
        ...
    ]
}
```

### Challenge

The main purpose of this exercise is to create a model that distinguishes between similar and different paths. Using the provided dataset, perform the following tasks:

* Create any additional variables you think necessary.
* Build a model with them to predict when two routes are identical or different (according to human evaluation).
* Evaluate the model’s performance.

Hint: Note that each journey may have more than one annotation, as it may have been viewed by more than one annotator, and these annotations may be different. Don’t forget that these are evaluations made by humans, each with their own criteria, and that they don’t always have to coincide about if a pair or estimate/real routes are similar or not. You will have to decide what target value to use for those pairs of routes where there is no match in the annotations.
---

Fuente: https://gist.github.com/jmontess/cd4bacd1b09301edf5f2e99ce07584d3 (descargado 2026-10-16)
