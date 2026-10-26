# Almaty Buildings

Most apartment blocks in Almaty were built in the USSR, and the USSR mostly built the same standard types of houses again and again. History showed that not all of these types are safe for an earthquake like Almaty can have. The project finds out the type of each building, to see which ones may be at risk.

There are about 9,000 apartment blocks in Almaty. I took about 7,000 of them from OpenStreetMap and downloaded Street View photos for around 1,100. About 900 I labelled by hand and used them for supervised learning.

While labelling I noticed that houses often look not like what they really are. Old houses look new after facade repair, and new houses that nobody looks after start to look old. Also I found that brick houses with bare brick almost don't exist anymore. Brick buildings are usually covered with square plates around one metre by one metre, or with other decoration.

After labelling I wanted to check how good today's LLMs are at telling the building type. In my test the OpenAI model was right on 90% of photos, and on 93% if we don't count a few small problems with the photos.

## Model

I took ResNet-50, which was already trained on about a million ordinary photos (ImageNet), and changed its last layer so it gives the three classes I need: panel, brick and monolith. Then I trained it more on my 464 labelled photos. The model gives the answer in percent, so we also see how sure it is. The main plus of the model is that it is fast and runs on my own laptop for free, so it can label the whole city.

<!-- TODO: how I split the data by district -->
<!-- TODO: results table (Codex, ResNet, CLIP) -->
<!-- TODO: what didn't work / limitations -->
<!-- TODO: next steps (map of the whole city, fault lines) -->

## Data source

Photos are from the Google Street View Static API. The photos are not in this repo, only the code and building coordinates. Buildings are from OpenStreetMap (© OpenStreetMap contributors).
