import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
    """Function to get the model based on the arguments.
    Actually the models available, only 3 dimensions, are:
        - UNETR
        - UNet
        - VNet
        - AttentionUNet
        - ResUNet
        - MedFormer
        - SegFormer
        - UNetPlusPlus
        - SwinUNETR
        - SAM3D
        - nnFormer: don't work properly TODO
        - VTUNet: don't work properly TODO
        - FCN_Net: don't work properly TODO
    TODO: Implement the other models, for example:
        - DeepLabV3
        - PSPNet


    Args:
        args (argparse.Namespace): Arguments from the command line.
        pretrain (bool, optional): Set to true if you use a pretrained model. Defaults to False.

    Raises:
        ValueError: No pretrain model available
        ValueError: Invalid dimension, should be '2d' or '3d'

    Returns:
        Model: The model object.
    """    
    
    if args.dimension == '3d':
        
        if args.model == 'unetr':
            from .dim3 import UNETR
            return UNETR(
                args.in_chan, 
                args.classes, 
                args.training_size, 
                feature_size=args.base_chan, 
                hidden_size=768, 
                mlp_dim=3072, 
                num_heads=12, 
                pos_embed='perceptron', 
                norm_name='instance', 
                res_block=True
                )
        
        elif args.model == "segformer":
            from .dim3 import SegFormer3D
            return SegFormer3D(
                in_channels=args.in_chan,
                sr_ratios=args.sr_ratios,
                embed_dims=args.embed_dims,
                patch_kernel_size=args.patch_kernel_size,
                patch_stride=args.patch_stride,
                patch_padding=args.patch_padding,
                mlp_ratios=args.mlp_ratios,
                num_heads=args.num_heads,
                depths=args.depths,
                decoder_head_embedding_dim=args.decoder_head_embedding_dim,
                num_classes=args.classes,
                decoder_dropout=args.decoder_dropout,
            )

        elif args.model == 'sam':
            from .dim3 import sam_model_registry3D
            return sam_model_registry3D[args.vit_name](checkpoint=None)
            raise print("\n[TODO] DONT WORK PROPERLY YET! TODO FIX IT!")
            return Sam3D(
                num_classes=args.classes, 
                ckpt=None, 
                image_size=args.crop_size, 
                vit_name=args.vit_name,
                num_modalities=args.in_chan, 
                do_ds=args.do_ds
                )

        elif args.model == 'dints':
            from .dim3 import DiNTS, TopologySearch, TopologyInstance
            # dints_space = TopologySearch(
            #     channel_mul=0.5,
            #     num_blocks=12,
            #     num_depths=4,
            #     use_downsample=True,
            #     device=args.device,
            # )
            dints_space = TopologyInstance(
                channel_mul=args.channel_mul,
                num_blocks=args.num_blocks,
                num_depths=args.num_depths,
                use_downsample=args.use_downsample,
                device=args.aug_device,
            )
            return DiNTS(
                    dints_space=dints_space,
                    in_channels=args.in_chan,
                    num_classes=args.classes,
                    act_name=args.act_name,
                    norm_name=("INSTANCE", {"affine": True}),
                    spatial_dims=args.spatial_dims,
                    use_downsample=args.use_downsample,
                    node_a=None,
                )

        elif args.model == 'resunet':
            from .dim3 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )

    
        elif args.model == 'vnet':
            from .dim3 import VNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return VNet(
                args.in_chan, 
                args.classes, 
                scale=args.downsample_scale, 
                baseChans=args.base_chan
                )

        elif args.model == 'attention_unet':
            from .dim3 import AttentionUNet
            return AttentionUNet(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )

        elif args.model == 'unet':
            from .dim3 import UNet
            return UNet(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )
        
        elif args.model == 'medformer':
            from .dim3 import MedFormer
            return MedFormer(
                args.in_chan, 
                args.classes, 
                args.base_chan, 
                map_size=args.map_size, 
                conv_block=args.conv_block, 
                conv_num=args.conv_num, 
                trans_num=args.trans_num, 
                num_heads=args.num_heads, 
                fusion_depth=args.fusion_depth, 
                fusion_dim=args.fusion_dim, 
                fusion_heads=args.fusion_heads, 
                expansion=args.expansion, 
                attn_drop=args.attn_drop, 
                proj_drop=args.proj_drop, 
                proj_type=args.proj_type, 
                norm=args.norm, 
                act=args.act, 
                kernel_size=args.kernel_size, 
                scale=args.down_scale, 
                aux_loss=args.aux_loss
                )
    
        elif args.model == 'unet++':
            from .dim3 import UNetPlusPlus
            return UNetPlusPlus(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )
        
        elif args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            return SwinUNETR(
                args.window_size, 
                args.in_chan, 
                args.classes, 
                feature_size=args.base_chan,
                num_heads=args.num_heads
                )
            
        elif args.model == 'nnformer':
            from .dim3 import nnFormer
            return nnFormer(
                args.window_size, 
                input_channels=args.in_chan, 
                num_classes=args.classes, 
                deep_supervision=args.aux_loss
                )
            
        elif args.model == 'vtunet':
            from .dim3 import VTUNet
            return VTUNet(
                args, 
                args.classes
                )

        elif args.model == 'fcn_net':
            from .dim3 import FCN_Net
            return FCN_Net(
                args.in_chan, 
                args.classes, 
                )
        
    elif args.dimension == '2d':
        if args.model == 'unet':                # check it
            from .dim2 import UNet
            return UNet(
                args.in_chan,
                args.classes
            )
        
        elif args.model == "segformer":         # check it
            from .dim2 import SegFormer     
            return SegFormer(
                args.in_chan,
                args.classes
            )

        

        # TODO: add swin_unetr
        elif args.model == "swin_unetr":
            from .dim2 import SwinUNETR
            return SwinUNETR(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_chans,
                out_channels=args.classes,         
                feature_size=args.embed_dim,
                patch_size=args.patch_size,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                dropout_path_rate=args.drop_path_rate,
                use_checkpoint=args.use_checkpoint,
                use_v2=False
            )

        # TODO: add unet++
        elif args.model == "unetpp":            # completed
            from .dim2 import BasicUNetPlusPlus
            return BasicUNetPlusPlus(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_chan,
                out_channels=args.classes,
                features=args.features,
                dropout=args.dropout
            )

        # TODO: add unetr
        elif args.model == "unetr":             # completed
            from .dim2 import UNETR
            return UNETR(
                in_channels=args.in_chan,
                out_channels=args.classes,
                img_size=args.roi_size,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                dropout_rate=args.dropout_rate,
                spatial_dims=args.spatial_dims,
                norm_name=args.norm_name
            )
        
        # TODO: add attention_unet
        elif args.model == "attention_unet":
            from .dim2 import AttentionUnet
            return AttentionUnet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.classes,
                channels=args.channels,
                strides=args.strides,
            )
        
        ##################################
        # That are bleeding-edge research models.
        # TODO: add unetr++
        elif args.model == "unetrpp":
            from .dim2 import UNETR_PP
            return UNETR_PP(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                dropout_rate=args.dropout_rate,
                depths=args.depths,
                dims=args.dims,
                conv_op=nn.Conv2d,
                do_ds=args.do_ds
            )

        # TODO: add uxlstm
        elif args.model == "uxlstm_bot":
            from .dim2 import UXlstmBot
            conv_op = nn.Conv2d if args.spatial_dims == 2 else nn.Conv3d
            norm_op = nn.InstanceNorm2d if args.spatial_dims == 2 else nn.InstanceNorm3d

            return UXlstmBot(
                input_channels=args.in_channels,
                n_stages=args.n_stages,
                features_per_stage=args.features_per_stage,
                conv_op=conv_op,
                kernel_sizes=args.kernel_sizes,
                strides=args.strides,
                n_conv_per_stage=args.n_conv_per_stage,
                num_classes=args.classes,
                n_conv_per_stage_decoder=args.n_conv_per_stage_decoder,
                conv_bias=True,
                norm_op=norm_op,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True},
                deep_supervision=False
            )
        ##################################

        # TODO: add medlsam2
        elif args.model == "medlsam":
            pass
        
        # TODO: add segmamba 2d
        elif args.model == "segmamba":
            from .dim2 import SegMamba
            return SegMamba(
                in_chans=args.in_chan,             
                out_chans=args.classes,       
                depths=args.depths,                
                feat_size=args.feat_size,          
                hidden_size=args.hidden_size,     
                norm_name=args.norm_name,          
                spatial_dims=args.spatial_dims                 
            )

        # TODO: add nnmamba 2d
        elif args.model == "nnmamba":
            from .dim2 import nnMambaSeg
            return nnMambaSeg(
                in_ch=args.in_chan,
                channels=args.base_chan,
                blocks=args.blocks,
                number_classes=args.classes
            )
    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')

