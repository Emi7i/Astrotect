# Astrotect  
Cilj ovog projekta je razvoj sistema za automatsku klasifikaciju astronomskih objekata (kometa, galaksija, nebula i globularnih skupova zvezda) na osnovu digitalnih slika svemira koje su snimili amateri. 

Problem se fokusira na precizno razdvajanje različitih tipova nebeskih tela koja na slikama često mogu izgledati slično zbog prisustva šuma, niske rezolucije ili velike udaljenosti objekata. Sistem treba da svakom detektovanom objektu na slici dodeli odgovarajuću klasu i zaokviri objekat.  
  
## Motivacija  
Amaterski astronomi svakodnevno prikupljaju ogromnu količinu slika svemira, ali često se suočavaju sa izazovom precizne identifikacije objekata bez adekvatnog stručnog znanja. Ovaj program bi ponudio rešenje za ta dva problema:  
 -  Automatsko klasifikovanje astronomskih objekata nad velikim brojem slika  
 -  Edukacija početnika i povećanje dostupnosti stručnih informacija  
   
## Skup podataka  
Koristiću COSMICA skup podataka. <br>  
Link: https://www.kaggle.com/datasets/piratinskii/astronomical-object-detection<br>  
Referentni rad: Piratinskii, E.; Rabaev, I. COSMICA: A Novel Dataset for Astronomical Object Detection with Evaluation Across Diverse Detection Architectures. J. Imaging 2025, 11, 184. https://doi.org/10.3390/jimaging11060184  
  
Dataset sadrži 5145 ručno anotiranih slika prikupljenih sa astronomy.ru foruma, gde korisnici dele svoje fotografije. Slike su podeljene u 4 klase: komete, galaksije, nebule i globularne skupove zvezda. Anotacije su dostupne u YOLO formatu (bounding box) sa obeleženom kategorijom objekata.   
  
## Metodologija  
Glavni alat za rešavanje problema biće Konvoluciona neuronska mreža (CNN):  
  
#### Pretprocesiranje 
Slike će biti konvertovane u grayscale, primeniću Gaussian blur ili median filtering za uklanjanje šuma. Koristiću neki Region Proposal algoritam (Selective Search) da mi vrati regije gde se verovatno nalazi objekat.  
  
### Arhitektura mreže  
 - Konvolucioni slojevi: Za ekstrakciju vizuelnih karakteristika poput ivica i oblika galaksija.  
 - Pooling slojevi (Max Pooling): Za smanjenje prostorne dimenzionalnosti i očuvanje najbitnijih informacija.  
 - Leaky ReLU aktivacija: Za uvođenje nelinearnosti u model.  
 - Dropout slojevi: Radi sprečavanja preprilagođavanja (overfitting-a).  
  
### Klasifikacija  
Na kraju mreže koristiću Fully-connected slojeve sa Softmax aktivacionom funkcijom na izlaznom sloju koja će dodeliti verovatnoću svakoj od četiri klase objekata (komete, galaksije, nebule, globularni skupovi).  
  
## Evaluacija  
Podela skupa podataka: trening (79%), validacioni (10.3%), testni (10.7%)  
  
### Metrike evaluacije:  
mAP (mean Average Precision) - meri preciznost detekcije i klasifikacije po svakoj klasi  
IoU (Intersection over Union) - mera preklapanja predviđenog i stvarnog bounding box-a, koristi se kao prag za određivanje tačnih detekcija  
Precision i Recall - za analizu pouzdanosti modela po svakoj od 4 klase objekata  
  
Opciono: k-fold unakrsna validacija iz razloga što skup podataka nije najbolje izbalansiran