# Generated by Django 4.2.15 on 2024-08-13 11:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mlapp', '0003_alter_chequedata_amount_words_alter_chequedata_date_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='chequedata',
            name='image_path',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
