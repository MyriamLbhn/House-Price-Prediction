# Generated by Django 4.1.7 on 2023-02-27 09:49

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Maison',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sqft_living', models.PositiveIntegerField()),
                ('bedrooms', models.PositiveIntegerField()),
                ('bathrooms', models.PositiveIntegerField()),
                ('sqft_lot', models.PositiveIntegerField()),
                ('floors', models.PositiveIntegerField()),
                ('sqft_basement', models.PositiveIntegerField()),
                ('sqft_above', models.PositiveIntegerField()),
                ('view', models.PositiveIntegerField(choices=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])),
                ('grade', models.PositiveIntegerField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])),
                ('condition', models.PositiveIntegerField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])),
                ('waterfront', models.BooleanField()),
                ('zipcode', models.CharField(max_length=5)),
                ('address', models.CharField(max_length=200)),
            ],
        ),
    ]
