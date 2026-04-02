from django.db import models
from person.models import ZipCode

class Place(models.Model):
    name=models.CharField(max_length=50)
    address=models.CharField(max_length=80)



# class Restaurant(models.Model):
#     zipcode=models.ForeignKey(
#         ZipCode,
#         on_delete=models.SET_NULL,
#         blank=True,
#         null=True,
#     )

class Restaurant(Place):
    serves_hot_dogs=models.BooleanField(default=False)
    serves_pizza=models.BooleanField(default=False)

class Person (models.Model):
    first_name=models.CharField(max_length=30)
    last_name=models.CharField(max_length=30)

class MyPerson(Person):
    
    class Meta:
        proxy=True # Définiton de la classe mandataire
    
    def do_something(self):
        pass

class OrderedPerson(Person):
    class Meta:
        ordering=["last_name"]
        proxy=True



